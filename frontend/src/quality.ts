/** Render quality. Phones (coarse pointer) default to lite; override with ?lite=0|1|2.
 *  0: bevelled cubes (108 tris) + N8AO (half res), dpr up to 2
 *  1: bevelled cubes, no AO, dpr 1          (phones)
 *  2: plain boxes (12 tris), no AO, dpr 1    (software GL / very old phones) */
const q = new URLSearchParams(location.search).get('lite')
export const LITE: 0 | 1 | 2 =
  q !== null ? (Math.min(2, Math.max(0, parseInt(q, 10) || 0)) as 0 | 1 | 2)
    : window.matchMedia?.('(pointer: coarse)').matches ? 1 : 0
