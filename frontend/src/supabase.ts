import { createClient, type Session } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string | undefined

/** null when the build has no Supabase settings: scanning still works, saving is hidden. */
export const supabase = url && key
  ? createClient(url, key, { auth: { flowType: 'pkce', detectSessionInUrl: true, persistSession: true } })
  : null

export async function accessToken(): Promise<string | null> {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()   // refreshes an expired token on its own
  return data.session?.access_token ?? null
}

export async function signInWithGoogle() {
  // Google -> Supabase callback -> back here (must be listed under Authentication -> URL Configuration).
  // prompt=select_account: Google always shows its account chooser. Without it, a browser still signed in to Google
  // goes straight back here, so after "Sign out" you could never pick a different account.
  await supabase?.auth.signInWithOAuth({
    provider: 'google',
    options: { redirectTo: `${location.origin}/`, queryParams: { prompt: 'select_account' } },
  })
}

export const signOut = async () => { await supabase?.auth.signOut() }

export interface User { id: string; email: string | null; name: string | null; avatar: string | null }

export const userOf = (s: Session | null): User | null => {
  const u = s?.user
  if (!u) return null
  const m = u.user_metadata ?? {}
  return { id: u.id, email: u.email ?? null, name: m.full_name ?? m.name ?? u.email ?? null, avatar: m.avatar_url ?? m.picture ?? null }
}
