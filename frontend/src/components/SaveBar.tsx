interface Props {
  saved: boolean           // the scan has a saved project
  name: string             // controlled by the parent (single source of truth for the project name)
  dirty: boolean
  busy: boolean
  signedIn: boolean
  onSave: () => void
  onNameChange: (name: string) => void
}

/** Explicit save: the first save stores the scan in your account, later saves update it. Shows unsaved changes. */
export function SaveBar(p: Props) {
  return (
    <div className="savebar panel">
      <input className="name-input" value={p.name} aria-label="project name" maxLength={80}
        onChange={(e) => p.onNameChange(e.target.value)} />
      <button className="brass-btn" disabled={p.busy || !p.name.trim()} onClick={p.onSave}>
        {p.busy ? 'Saving…' : p.saved ? 'Save' : p.signedIn ? 'Save project' : 'Sign in & save'}
      </button>
      <span className={`save-state ${p.dirty ? 'dirty' : p.saved ? 'clean' : ''}`}>
        {!p.saved ? 'draft (not saved)' : p.dirty ? '● unsaved changes' : '✓ saved'}
      </span>
    </div>
  )
}
