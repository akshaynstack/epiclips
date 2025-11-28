# Security Implementation Plan: API ↔ Genesis Communication

> **Created**: 2025-11-28
> **Status**: ✅ Implemented
> **Priority**: High (Required before AWS Cloud Map deployment)

---

## Overview

This plan covers implementing mutual authentication between viewcreator-api and viewcreator-genesis:

1. **API Key Authentication (API → Genesis)**: Secure job submission requests
2. **HMAC Webhook Signatures (Genesis → API)**: Secure webhook callbacks

---

## Environment Variables Summary

### viewcreator-api/.env

Add these two environment variables:

```
GENESIS_API_KEY=<your-secure-random-key>
GENESIS_WEBHOOK_SECRET=<your-secure-random-key>
```

### viewcreator-genesis (docker-compose.yml or .env)

Add the same two environment variables with **identical values**:

```
GENESIS_API_KEY=<same-key-as-api>
GENESIS_WEBHOOK_SECRET=<same-key-as-api>
```

### Generating Secure Keys

```bash
# Generate a secure random key (run twice for both variables)
openssl rand -hex 32
```

---

## Phase 1: API Key Authentication (API → Genesis)

**Purpose**: Ensure only the viewcreator-api can submit jobs to genesis.

### 1.1 viewcreator-api Changes

**File**: `src/modules/ai-clipping-agent/services/clipping-worker-client.service.ts`

**Changes**:
- Read `GENESIS_API_KEY` from config in constructor
- Add `X-Genesis-API-Key` header to all outgoing requests
- Log warning on startup if key is not configured (dev mode)

**Implementation**:
```typescript
// In constructor:
this.genesisApiKey = this.configService.get<string>('GENESIS_API_KEY')?.trim() || null;

// In axios client creation:
headers: {
  'Content-Type': 'application/json',
  ...(this.genesisApiKey && { 'X-Genesis-API-Key': this.genesisApiKey }),
}
```

### 1.2 viewcreator-genesis Changes

**File**: Create `app/middleware/api_key_auth.py`

**Changes**:
- Create FastAPI dependency for API key validation
- Read `GENESIS_API_KEY` from environment
- Return 401 Unauthorized if key is missing or invalid
- Skip validation if key is not configured (dev mode)

**File**: `app/routers/ai_clipping.py`

**Changes**:
- Apply API key dependency to job submission endpoints:
  - `POST /ai-clipping/jobs`
  - `DELETE /ai-clipping/jobs/{job_id}`
- Keep health and status endpoints public (for monitoring)

---

## Phase 2: HMAC Webhook Signatures (Genesis → API)

**Purpose**: Ensure webhook payloads are authentic and haven't been tampered with.

### 2.1 viewcreator-genesis Changes

**File**: `app/services/webhook_service.py`

**Changes**:
- Read `GENESIS_WEBHOOK_SECRET` from environment
- Implement HMAC-SHA256 signature generation
- Add `X-Genesis-Webhook-Signature` header with format: `sha256=<hex-signature>`
- Sign the JSON payload body

**Signature Algorithm**:
```python
import hmac
import hashlib

def sign_payload(payload_json: str, secret: str) -> str:
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_json.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"
```

### 2.2 viewcreator-api Changes

**File**: `src/modules/webhooks/webhooks.controller.ts`

**Changes**:
- Rename `WEBHOOK_SECRET` to `GENESIS_WEBHOOK_SECRET` (update config read)
- Implement proper HMAC-SHA256 verification
- Read raw body for signature verification (NestJS middleware needed)
- Compare signatures using timing-safe comparison

**File**: Create raw body middleware or use `@RawBody()` decorator

**Signature Verification**:
```typescript
import * as crypto from 'crypto';

function verifySignature(payload: string, signature: string, secret: string): boolean {
  const expectedSignature = 'sha256=' + crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');

  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );
}
```

---

## File Changes Summary

### viewcreator-api

| File | Change |
|------|--------|
| `src/modules/ai-clipping-agent/services/clipping-worker-client.service.ts` | Add `GENESIS_API_KEY` header |
| `src/modules/webhooks/webhooks.controller.ts` | Implement HMAC verification, rename secret |
| `src/modules/webhooks/webhooks.module.ts` | Add raw body parsing if needed |
| `.env.example` | Add `GENESIS_API_KEY` and `GENESIS_WEBHOOK_SECRET` |

### viewcreator-genesis

| File | Change |
|------|--------|
| `app/middleware/api_key_auth.py` | **NEW** - API key validation dependency |
| `app/services/webhook_service.py` | Add HMAC signing |
| `app/routers/ai_clipping.py` | Apply API key auth to protected routes |
| `app/config.py` | Add new settings for secrets |
| `docker-compose.yml` | Add environment variables |

---

## Implementation Order

1. **Phase 1A**: Add API key to viewcreator-api (sends header, but genesis ignores it)
2. **Phase 1B**: Add API key validation to genesis (now enforced)
3. **Phase 2A**: Add HMAC signing to genesis (sends signature, but API ignores it)
4. **Phase 2B**: Add HMAC verification to API (now enforced)

This order ensures no breaking changes during rollout.

---

## Testing Checklist

### API Key Authentication
- [ ] API sends `X-Genesis-API-Key` header on job submission
- [ ] Genesis accepts requests with valid API key
- [ ] Genesis rejects requests with missing API key (401)
- [ ] Genesis rejects requests with invalid API key (401)
- [ ] Health endpoint remains accessible without API key

### HMAC Webhook Signatures
- [ ] Genesis signs all webhook payloads
- [ ] API verifies signatures when secret is configured
- [ ] API rejects webhooks with invalid signatures (401)
- [ ] API rejects webhooks with missing signatures (401)
- [ ] Signature is computed over exact JSON payload

---

## Security Considerations

### AWS Cloud Map Deployment

When deployed to AWS with Cloud Map:
1. **Network Isolation**: Keep genesis in private subnet (no public IP)
2. **Service Discovery**: API discovers genesis via Cloud Map DNS
3. **Defense in Depth**: API key provides authentication even within VPC
4. **Audit Trail**: Log all authentication failures for security monitoring

### Key Rotation

To rotate keys without downtime:
1. Configure new key in genesis (accept both old and new)
2. Update API to use new key
3. Remove old key from genesis

---

## Rollback Plan

If issues occur:
1. Set `GENESIS_API_KEY` to empty string on both services (disables auth)
2. Set `GENESIS_WEBHOOK_SECRET` to empty string on both services (disables verification)
3. Investigate and fix issues
4. Re-enable with corrected configuration
