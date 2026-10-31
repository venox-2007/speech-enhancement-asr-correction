param([switch]$ExportBackup)
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DemoPresentationFonts {
 [DllImport("gdi32.dll", CharSet=CharSet.Unicode)] public static extern int AddFontResourceEx(string file, uint flags, IntPtr reserved);
 [DllImport("gdi32.dll", CharSet=CharSet.Unicode)] public static extern bool RemoveFontResourceEx(string file, uint flags, IntPtr reserved);
}
'@
$loadedFonts=@()
$pp=$null
$deck=$null
try {
 foreach($name in @('Epilogue.ttf','Epilogue-Bold.ttf','Fraunces-Medium.ttf')){
  $fontPath=Join-Path $PSScriptRoot ('assets\fonts\'+$name)
  if([DemoPresentationFonts]::AddFontResourceEx($fontPath,0,[IntPtr]::Zero) -eq 0){throw ('Could not load '+$name)}
  $loadedFonts+=$fontPath
 }
 $pp=New-Object -ComObject PowerPoint.Application
 $pptPath=Join-Path $PSScriptRoot 'Speech-Enhancement-ASR-SUBMISSION.pptx'
 if($ExportBackup){
  $deck=$pp.Presentations.Open($pptPath,-1,0,0)
  $exportFolder=Join-Path (Split-Path $PSScriptRoot -Parent) 'results\ppt_build\submission-native-slides'
  $deck.Export($exportFolder,'PNG',1280,720)

  Write-Output ('Exported '+$deck.Slides.Count+' slides through PowerPoint.')
  $deck.Close()
  $deck=$null
 } else {
  $pp.Visible=-1
  $deck=$pp.Presentations.Open($pptPath,0,0,-1)
  # Keep temporary font resources available until this presentation closes.
  while($true){
   try {$null=$deck.Slides.Count} catch {break}
   Start-Sleep -Seconds 2
  }
 }
} finally {
 # Never close a presentation that the user is editing.
 if($ExportBackup -and $null -ne $deck){$deck.Close()}
 if($null -ne $pp){try {if($pp.Presentations.Count -eq 0){$pp.Quit()}} catch {}}
 foreach($fontPath in $loadedFonts){[DemoPresentationFonts]::RemoveFontResourceEx($fontPath,0,[IntPtr]::Zero) | Out-Null}
}
