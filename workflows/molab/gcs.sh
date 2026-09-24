#!/bin/bash
# gcs.sh
# Purpose: Talk to the private molab bucket without gcloud. Sourced by
#   setup_molab.sh (downloads) and sync_to_gcs.sh (uploads); defines functions
#   only and runs nothing.
#
# Why not gcloud: there is none on the box, and installing it for a handful of
# object reads and writes is a large download onto a disk that counts. An OAuth
# token is minted from the service-account key with openssl instead (the JWT
# bearer flow), and objects move through the JSON API with curl. The key
# reaches openssl through a pipe (process substitution) and never touches
# disk; the token reaches curl the same way, so neither shows in `ps`.
#
# Why GCS_SCOPE: setup only reads, so it asks for devstorage.read_only; the
# sync writes and asks for read_write. A token never carries more than the
# caller needs, whatever the service account itself is allowed.
#
# Input:  GCP_BUCKET, GCP_SA_JSON (the service-account key as inline JSON),
#         GCS_SCOPE (read_only by default), and a log() from the caller
# Output: functions: gcs_token, gcs_meta, gcs_list, gcs_fetch, gcs_put, sa_field
# Usage:  source workflows/molab/gcs.sh; GCS_TOKEN=$(gcs_token)
# Prerequisites: curl, openssl, python3

# shellcheck disable=SC2218  # 0.11.0 false positive (see CLAUDE.md): every function is defined before use
# shellcheck disable=SC2034  # GCS_PUT is read by the caller (sync_to_gcs.sh)

GCS_SCOPE="${GCS_SCOPE:-read_only}"

sa_field() {
    printf '%s' "${GCP_SA_JSON}" \
        | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin)[sys.argv[1]])' "$1"
}
b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

gcs_token() {
    local now header claims unsigned sig
    now=$(date +%s)
    header=$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)
    claims=$(printf '{"iss":"%s","scope":"https://www.googleapis.com/auth/devstorage.%s","aud":"https://oauth2.googleapis.com/token","iat":%d,"exp":%d}' \
        "$(sa_field client_email)" "${GCS_SCOPE}" "${now}" "$((now + 3600))" | b64url)
    unsigned="${header}.${claims}"
    sig=$(printf '%s' "${unsigned}" | openssl dgst -sha256 -sign <(sa_field private_key) | b64url)
    curl -fsS https://oauth2.googleapis.com/token \
        --data-urlencode "grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer" \
        --data-urlencode "assertion@"<(printf '%s' "${unsigned}.${sig}") \
        | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin)["access_token"])'
}

GCS_TOKEN=""
auth_header() { printf 'Authorization: Bearer %s\n' "${GCS_TOKEN}"; }
urlenc() { python3 -c 'import sys, urllib.parse; sys.stdout.write(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
gcs_api() { curl -fsS -H @<(auth_header) "https://storage.googleapis.com/storage/v1/b/${GCP_BUCKET}/o$1"; }

# "<size> <md5>" of one object; md5 is "-" for composite objects, which have
# none. Fails (non-zero, no output) when the object does not exist.
gcs_meta() {
    gcs_api "/$(urlenc "$1")?fields=size,md5Hash" 2>/dev/null \
        | python3 -c 'import json, sys; j = json.load(sys.stdin); print(j["size"], j.get("md5Hash") or "-")' 2>/dev/null
}

# Every object name under a prefix, one per line, directory placeholders skipped.
gcs_list() {
    local page="" out
    while :; do
        out=$(gcs_api "?prefix=$(urlenc "$1")&fields=items(name),nextPageToken${page:+&pageToken=$(urlenc "${page}")}")
        printf '%s' "${out}" | python3 -c 'import json, sys; [print(i["name"]) for i in json.load(sys.stdin).get("items", []) if not i["name"].endswith("/")]'
        page=$(printf '%s' "${out}" | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin).get("nextPageToken", ""))')
        [[ -n "${page}" ]] || break
    done
}

md5_b64() { openssl dgst -md5 -binary "$1" | base64; }

# Does <file> match the bucket's <size> and <md5>? md5 "-" means size only.
matches() {
    [[ -f "$1" ]] && [[ "$(stat -c %s "$1")" == "$2" ]] || return 1
    [[ "$3" == "-" ]] || [[ "$(md5_b64 "$1")" == "$3" ]]
}

# Download <object> to <dest> unless it is already there and intact. Resumes a
# partial <dest>.part; mints a fresh token between attempts because each lasts
# an hour, which a slow 8 GB download can outlive.
gcs_fetch() {
    local obj="$1" dest="$2" size md5 attempt
    read -r size md5 < <(gcs_meta "${obj}") || true
    [[ -n "${size}" ]] || { echo "ERROR: gs://${GCP_BUCKET}/${obj} not found or not readable" >&2; return 1; }
    if matches "${dest}" "${size}" "${md5}"; then
        log "present: ${dest}"
        return 0
    fi
    rm -f "${dest}"
    mkdir -p "$(dirname "${dest}")"
    [[ -f "${dest}.part" ]] && (( $(stat -c %s "${dest}.part") > size )) && rm -f "${dest}.part"
    log "downloading gs://${GCP_BUCKET}/${obj} ($(numfmt --to=iec "${size}"))"
    for attempt in 1 2 3; do
        (( attempt == 1 )) || GCS_TOKEN=$(gcs_token)
        curl -fL --retry 5 --retry-all-errors -sS -C - -H @<(auth_header) -o "${dest}.part" \
            "https://storage.googleapis.com/storage/v1/b/${GCP_BUCKET}/o/$(urlenc "${obj}")?alt=media" && break
        (( attempt < 3 )) || { echo "ERROR: download of ${obj} failed" >&2; return 1; }
    done
    matches "${dest}.part" "${size}" "${md5}" \
        || { echo "ERROR: ${dest}.part does not match gs://${GCP_BUCKET}/${obj} (size ${size}, md5 ${md5})" >&2; rm -f "${dest}.part"; return 1; }
    mv "${dest}.part" "${dest}"
}

# Upload <file> to <object> unless the bucket already holds identical bytes,
# so re-running after every step moves only what changed. Streams the file
# (curl -T), never buffers it; checks the md5 GCS computed against the local
# one. Sets GCS_PUT to "uploaded" or "unchanged" for the caller to count.
# Never deletes anything remote.
gcs_put() {
    local file="$1" obj="$2" size md5 local_md5 remote_md5 attempt
    GCS_PUT=""
    if read -r size md5 < <(gcs_meta "${obj}") && [[ -n "${size}" ]] && matches "${file}" "${size}" "${md5}"; then
        GCS_PUT="unchanged"
        return 0
    fi
    local_md5=$(md5_b64 "${file}")
    for attempt in 1 2 3; do
        (( attempt == 1 )) || GCS_TOKEN=$(gcs_token)
        remote_md5=$(curl -fsS --retry 5 --retry-all-errors -X POST -T "${file}" \
            -H @<(auth_header) -H "Content-Type: application/octet-stream" \
            "https://storage.googleapis.com/upload/storage/v1/b/${GCP_BUCKET}/o?uploadType=media&name=$(urlenc "${obj}")&fields=md5Hash" \
            | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin).get("md5Hash", ""))') && break
        (( attempt < 3 )) || { echo "ERROR: upload of ${file} to gs://${GCP_BUCKET}/${obj} failed" >&2; return 1; }
    done
    [[ "${remote_md5}" == "${local_md5}" ]] \
        || { echo "ERROR: gs://${GCP_BUCKET}/${obj} md5 ${remote_md5} != local ${local_md5}" >&2; return 1; }
    GCS_PUT="uploaded"
}
