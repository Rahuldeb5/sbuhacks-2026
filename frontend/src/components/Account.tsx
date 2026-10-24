import { signInWithGoogle, signOut, supabase, type User } from '../supabase'

export function Account({ user }: { user: User | null }) {
  if (!supabase) return null
  if (!user) {
    return <button className="steel-btn account-signin" onClick={() => signInWithGoogle()}>Sign in with Google</button>
  }
  return (
    <div className="account">
      {user.avatar ? (
        <img className="avatar" src={user.avatar} alt="" referrerPolicy="no-referrer" />
      ) : (
        <span className="avatar avatar-initial">{user.name?.[0] ?? '?'}</span>
      )}
      <span className="account-name" title={user.email ?? ''}>{user.name}</span>
      <button className="steel-btn" onClick={() => signOut()}>Sign out</button>
    </div>
  )
}
