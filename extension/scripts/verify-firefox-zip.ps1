Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
$zip = 'C:\Users\ansha\Desktop\PageGlance-Releases\PageGlance-Firefox-v0.1.0.zip'
$arc = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
  Write-Output "=== ENTRIES ($($arc.Entries.Count)) ==="
  foreach ($e in $arc.Entries) { Write-Output ("  [" + $e.FullName + "]") }

  Write-Output "=== BACKSLASH CHECK ==="
  $bs = $arc.Entries | Where-Object { $_.FullName.Contains('\') }
  if ($bs.Count -eq 0) { Write-Output "  PASS: no entry contains backslash" } else { Write-Output ("  FAIL: " + $bs.Count + " backslash entries") }

  Write-Output "=== ROOT CHECK ==="
  $root = $arc.Entries | Where-Object { -not $_.FullName.Contains('/') } | ForEach-Object { $_.FullName }
  Write-Output ("  root entries: " + ($root -join ', '))

  Write-Output "=== MANIFEST CONTENT ==="
  $mf = $arc.GetEntry('manifest.json')
  $r = New-Object System.IO.StreamReader($mf.Open())
  $txt = $r.ReadToEnd(); $r.Close()
  Write-Output $txt

  Write-Output "=== URL CHECK (search all text entries) ==="
  foreach ($e in $arc.Entries) {
    if ($e.FullName -match '\.(js|css|html|json)$') {
      $rr = New-Object System.IO.StreamReader($e.Open())
      $c = $rr.ReadToEnd(); $rr.Close()
      if ($c -match 'ansh9057-pageglance-backend\.hf\.space') { Write-Output ("  PROD URL in: " + $e.FullName) }
      if ($c -match '127\.0\.0\.1|localhost|DEV_SENTINEL') { Write-Output ("  LOCALHOST/DEV in: " + $e.FullName) }
    }
  }

  Write-Output "=== SIZE ==="
  Write-Output ("  bytes: " + (Get-Item $zip).Length)
} finally { $arc.Dispose() }
