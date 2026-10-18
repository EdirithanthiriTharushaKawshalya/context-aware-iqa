; Build the application using desktop.spec first, then compile with Inno Setup.
[Setup]
; Preserve the original uninstall identity when updating an existing install.
AppId=Context-Aware IQA
AppName=Context Aware IQA
AppVersion=0.5.2
DefaultDirName={localappdata}\Programs\ContextAwareIQA
DefaultGroupName=Context Aware IQA
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=ContextAwareIQA-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\src\images\icon.ico

[Files]
Source: "..\dist\ContextAwareIQA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

Source: "..\src\images\icon.ico"; DestDir: "{app}\src\images"; Flags: ignoreversion

[Icons]
Name: "{group}\Context Aware IQA"; Filename: "{app}\ContextAwareIQA.exe"; IconFilename: "{app}\src\images\icon.ico"
Name: "{autodesktop}\Context Aware IQA"; Filename: "{app}\ContextAwareIQA.exe"; IconFilename: "{app}\src\images\icon.ico"

[InstallDelete]
Type: files; Name: "{autodesktop}\Context-Aware IQA.lnk"
Type: files; Name: "{group}\Context-Aware IQA.lnk"

[Run]
Filename: "{app}\ContextAwareIQA.exe"; Description: "Launch Context Aware IQA"; Flags: nowait postinstall skipifsilent
