-- Steel & Steam: saved projects + scan storage.
--
-- The backend uses the secret key (bypasses RLS) and checks the user's JWT itself; the policies below are the
-- second line of defence and also let the browser read its own rows/files directly if we ever want that.

-- ---------------------------------------------------------------- projects (one row per saved scan)
create table if not exists public.projects (
  id              bigint generated always as identity primary key,
  user_id         uuid not null references auth.users (id) on delete cascade,
  scan_id         text not null unique check (scan_id ~ '^\d{8}-\d{6}-[0-9a-f]{6}$'),
  name            text not null check (char_length(name) between 1 and 80),
  method          text not null check (method in ('mirror', 'revolve', 'inflate')),
  thickness       real not null default 1.0 check (thickness between 0.25 and 2.5),
  plane_offset_cm real not null default 0.0,
  known_dim       jsonb,                                  -- e.g. {"h": 6.0}; null = estimate / marker
  scale_source    text not null default 'estimate',
  dims_cm         jsonb,                                  -- {"w","h","d"} of the saved model, for the archive cards
  description     jsonb,                                  -- the VLM's notes at save time
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index if not exists projects_user_updated on public.projects (user_id, updated_at desc);

create or replace function public.touch_updated_at() returns trigger
language plpgsql set search_path = '' as $$
begin
  new.updated_at := now();
  return new;
end $$;

drop trigger if exists projects_touch on public.projects;
create trigger projects_touch before update on public.projects
  for each row execute function public.touch_updated_at();

alter table public.projects enable row level security;

drop policy if exists "own projects: select" on public.projects;
drop policy if exists "own projects: insert" on public.projects;
drop policy if exists "own projects: update" on public.projects;
drop policy if exists "own projects: delete" on public.projects;
create policy "own projects: select" on public.projects for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "own projects: insert" on public.projects for insert to authenticated
  with check ((select auth.uid()) = user_id);
create policy "own projects: update" on public.projects for update to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "own projects: delete" on public.projects for delete to authenticated
  using ((select auth.uid()) = user_id);

-- ---------------------------------------------------------------- storage: private bucket, one folder per user
-- objects live at  scans/<user_id>/<scan_id>/<file>   (photo.jpg, depth_f.npy, masks.npz, meta.json, ...)
insert into storage.buckets (id, name, public, file_size_limit)
values ('scans', 'scans', false, 52428800)                  -- 50 MB per file (a scan folder is a few MB in total)
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;

drop policy if exists "own scan files: select" on storage.objects;
drop policy if exists "own scan files: insert" on storage.objects;
drop policy if exists "own scan files: update" on storage.objects;
drop policy if exists "own scan files: delete" on storage.objects;
create policy "own scan files: select" on storage.objects for select to authenticated
  using (bucket_id = 'scans' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own scan files: insert" on storage.objects for insert to authenticated
  with check (bucket_id = 'scans' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own scan files: update" on storage.objects for update to authenticated
  using (bucket_id = 'scans' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "own scan files: delete" on storage.objects for delete to authenticated
  using (bucket_id = 'scans' and (storage.foldername(name))[1] = (select auth.uid())::text);
