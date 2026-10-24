"""GPU arbitration between Depth Anything (in-process, torch) and the VLM (LM Studio, separate process).

GPU_MODE (env):
  coresident (default): the VLM stays loaded next to DA. At startup it is (re)loaded with the smallest settings
      that work: context 4096, parallel 1, flash attention, KV on GPU (measured: VLM +8.3 GB, DA fp16 +2.1 GB,
      peak 11.0 of 12.2 GB).
  swap: before depth, unload the VLM; after depth, reload it; then describe.

Locks (asyncio, on the server's event loop):
  gpu_lock  - held by depth inference AND by VLM load/unload, so depth never runs during a VLM (un)load.
              In swap mode the whole describe (reload + chat) also holds it, so the next scan's unload can't
              pull the model out from under a running request.
  vlm_lock  - one VLM request at a time.
Thread code (the scan runs in a worker thread) uses the *_sync wrappers, which hop onto the loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

log = logging.getLogger("gpu_manager")

LMSTUDIO = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3-vl-8b")
VLM_CTX = int(os.environ.get("VLM_CONTEXT", "4096"))
VLM_PARALLEL = int(os.environ.get("VLM_PARALLEL", "1"))


class GpuManager:
    def __init__(self, mode: str | None = None):
        self.mode = (mode or os.environ.get("GPU_MODE", "coresident")).lower()
        if self.mode not in ("coresident", "swap"):
            raise ValueError(f"GPU_MODE must be coresident|swap, got {self.mode}")
        self.gpu_lock = asyncio.Lock()
        self.vlm_lock = asyncio.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.instance_id: str | None = None
        self.available = False          # LM Studio reachable + model present
        self.stats = {"loads": [], "unloads": []}
        self.client = httpx.AsyncClient(timeout=120)

    # ---------------------------------------------------------------- LM Studio
    async def _instances(self) -> list[dict]:
        r = await self.client.get(f"{LMSTUDIO}/api/v1/models", timeout=5)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("key") == VLM_MODEL:
                self.available = True
                return m.get("loaded_instances", [])
        return []

    async def _unload_all(self):
        for inst in await self._instances():
            t = time.perf_counter()
            await self.client.post(f"{LMSTUDIO}/api/v1/models/unload", json={"instance_id": inst["id"]})
            self.stats["unloads"].append(round(time.perf_counter() - t, 2))
        self.instance_id = None

    async def _load(self):
        t = time.perf_counter()
        r = await self.client.post(f"{LMSTUDIO}/api/v1/models/load", json={
            "model": VLM_MODEL, "context_length": VLM_CTX, "parallel": VLM_PARALLEL, "flash_attention": True,
            "offload_kv_cache_to_gpu": True, "echo_load_config": True}, timeout=180)
        r.raise_for_status()
        j = r.json()
        self.instance_id = j["instance_id"]
        self.stats["loads"].append(round(time.perf_counter() - t, 2))
        log.info("VLM loaded %s in %.1fs config=%s", self.instance_id, time.perf_counter() - t, j.get("load_config"))

    async def _ensure_loaded(self):
        """Caller holds gpu_lock. Reuse an instance with the right config; otherwise unload all and load ours
        (a load always ADDS an instance, so never load on top of an existing one)."""
        insts = await self._instances()
        ok = [i for i in insts if i["config"].get("context_length") == VLM_CTX and i["config"].get("parallel") == VLM_PARALLEL]
        if ok and len(insts) == 1:
            self.instance_id = ok[0]["id"]
            return
        if insts:
            await self._unload_all()
        await self._load()

    async def startup(self):
        self.loop = asyncio.get_running_loop()
        try:
            if self.mode == "coresident":
                async with self.gpu_lock:
                    await self._ensure_loaded()
            else:
                await self._instances()
        except Exception as e:  # noqa: BLE001  - VLM is optional; scans must still work
            self.available = False
            log.warning("VLM unavailable (%s); descriptions will use the fallback", e)

    # ---------------------------------------------------------------- depth
    async def run_depth(self, fn, *args):
        """Run a GPU depth job. Swap mode: make sure the VLM is unloaded first."""
        async with self.gpu_lock:
            if self.mode == "swap" and self.available:
                try:
                    if await self._instances():
                        await self._unload_all()
                except Exception as e:  # noqa: BLE001
                    log.warning("VLM unload failed: %s", e)
            return await asyncio.to_thread(fn, *args)

    def run_depth_sync(self, fn, *args):
        if self.loop is None:
            return fn(*args)
        return asyncio.run_coroutine_threadsafe(self.run_depth(fn, *args), self.loop).result()

    # ---------------------------------------------------------------- VLM
    async def chat(self, body: dict, timeout: float) -> dict:
        """One VLM request (serialized). Swap mode reloads the model first, holding gpu_lock throughout."""
        if not self.available:
            raise RuntimeError("VLM unavailable")
        async with self.vlm_lock:
            if self.mode == "swap":
                async with self.gpu_lock:
                    await self._ensure_loaded()
                    return await self._post_chat(body, timeout)
            if self.instance_id is None:
                async with self.gpu_lock:
                    await self._ensure_loaded()
            return await self._post_chat(body, timeout)

    async def _post_chat(self, body, timeout):
        body = {**body, "model": self.instance_id or VLM_MODEL}
        r = await self.client.post(f"{LMSTUDIO}/api/v1/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def status(self) -> dict:
        return {"mode": self.mode, "vlm_available": self.available, "instance_id": self.instance_id,
                "model": VLM_MODEL, "context_length": VLM_CTX, "parallel": VLM_PARALLEL, **self.stats}
