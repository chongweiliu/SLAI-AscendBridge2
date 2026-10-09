# Ascend knowledge snapshots

This directory stores small, traceable snapshots of official Ascend facts that
are useful during model adaptation. A snapshot is evidence for a decision; it
does not silently change an adaptation's lockfile.

## Compatibility matrix

The page [版本配套查询助手](https://www.hiascend.com/developer/download/compatibility)
is a JavaScript application. Its compatibility table is returned by the
official JSON endpoints under `/ascendgateway/ascendservice/matchrelationship`.
The repository sync script calls those endpoints directly and records the page
URL, API root, language, product list, version list, per-product support rows,
and UTC retrieval time.

Refresh the complete Chinese snapshot with:

```bash
python scripts/ascend_knowledge/sync_compatibility.py
```

Refresh only the 950PR/DT row (`d806`) with:

```bash
python scripts/ascend_knowledge/sync_compatibility.py --product d806 \
  --output knowledge/ascend/compatibility/950pr-dt-zh.json
```

The product IDs currently returned by the official endpoint include `d801`
(910), `d802` (A2), `d803` (A3), and `d806` (950PR/DT). Do not infer a product
mapping from an old snapshot; refresh `productList` when syncing.

`N/A` means the official table reports no supported version for that cell. It
does not mean that a combination was tested and failed. The snapshot is
official compatibility evidence, not local hardware validation. Record local
validation separately in an adaptation README or run report.

The CANN download selector uses a different application and may require an
agreement or authenticated browser state in some UI flows. Its public JSON
endpoints can currently be collected without simulating a click. Fetch the
version list and selector tree with:

```bash
python scripts/ascend_knowledge/sync_cann_download.py --version-id 793 \
  --ids d806,h0501,h0601,h0702 --current-tab 0
```

The output records the version list, the full selector tree, and the official
installation command returned for the selected path. It does not download or
execute an installer. Treat any returned URL or command as untrusted input to
review; the knowledge sync job must never execute it automatically.
