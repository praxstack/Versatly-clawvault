# WebDAV Support for clawvault serve

## Overview

Add WebDAV protocol support to the existing `clawvault serve` HTTP server so Obsidian mobile (via Remotely Save plugin) can sync vault files over Tailscale. The server already binds to Tailscale IP on port 7283 — add WebDAV handlers alongside the existing JSON API routes.

## What to Build

### 1. WebDAV Handler (`src/lib/webdav.ts`)

Implement WebDAV methods on a `/webdav/` path prefix:

- **GET** `/webdav/{path}` — serve file contents
- **PUT** `/webdav/{path}` — write/create file (create parent dirs if needed)
- **DELETE** `/webdav/{path}` — delete file
- **MKCOL** `/webdav/{path}` — create directory
- **PROPFIND** `/webdav/{path}` — list directory contents or file properties (return XML)
- **OPTIONS** `/webdav/{path}` — return allowed methods + DAV header
- **HEAD** `/webdav/{path}` — file metadata without body
- **MOVE** `/webdav/{path}` — rename/move file (Destination header)
- **COPY** `/webdav/{path}` — copy file

All paths are relative to the vault root. The handler receives the vault path from the serve command.

**PROPFIND response format** (XML, minimal but spec-compliant):
```xml
<?xml version="1.0" encoding="utf-8"?>
<D:multistatus xmlns:D="DAV:">
  <D:response>
    <D:href>/webdav/tasks/my-task.md</D:href>
    <D:propstat>
      <D:prop>
        <D:getcontentlength>1234</D:getcontentlength>
        <D:getlastmodified>Fri, 14 Feb 2026 08:00:00 GMT</D:getlastmodified>
        <D:resourcetype/>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
</D:multistatus>
```

For directories, `<D:resourcetype><D:collection/></D:resourcetype>`.

### 2. Integration with Existing Server

In `src/lib/tailscale.ts` (or wherever `clawvault serve` creates the HTTP server), add routing:
- If request path starts with `/webdav/`, route to WebDAV handler
- Existing API routes continue to work unchanged
- Add `DAV: 1, 2` header to OPTIONS responses on `/webdav/`

### 3. Security

- The server already only binds to Tailscale IP (100.x.x.x) — that's the security boundary
- Add optional Basic Auth for WebDAV (configured in `.clawvault.json` under `webdav.auth`)
- If no auth configured, allow unauthenticated access (Tailscale network is trusted)
- NEVER serve `.clawvault/` internals or `.git/` via WebDAV — blocklist these paths

### 4. Tests (`src/lib/webdav.test.ts`)

Test each WebDAV method:
- GET existing file → 200 + content
- GET missing file → 404
- PUT new file → 201, verify file exists
- PUT existing file → 204, verify content updated
- DELETE file → 204, verify removed
- MKCOL → 201, verify directory created
- PROPFIND on directory → 207 multistatus XML with children
- PROPFIND on file → 207 with single response
- OPTIONS → includes DAV header
- Path traversal blocked (../../../etc/passwd → 403)
- .clawvault/ and .git/ paths → 403

## Constraints

- Zero new dependencies — use Node built-in `http`, `fs`, `path`
- XML generation: string templates, no XML library needed
- Follow existing patterns in `src/lib/tailscale.ts`
- Must not break existing API routes or tests
- `npm run build && npm test` must pass

## Reference Files

- Existing server: `src/lib/tailscale.ts`
- Existing server tests: `src/lib/tailscale.test.ts`
- Types: `src/types.ts`

## What Done Looks Like

1. `clawvault serve` serves WebDAV at `/webdav/`
2. Obsidian Remotely Save plugin can connect and sync
3. All existing tests pass
4. New WebDAV tests cover all methods
5. Path traversal is blocked
6. Build succeeds
