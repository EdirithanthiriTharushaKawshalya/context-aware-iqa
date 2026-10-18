; Build the application using desktop.spec first, then compile with Inno Setup.
[Setup]
AppName=Context-Aware IQA
AppVersion=0.3.0
DefaultDirName={localappdata}\Programs\ContextAwareIQA
DefaultGroupName=Context-Aware IQA
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=ContextAwareIQA-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "..\dist\ContextAwareIQA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Context-Aware IQA"; Filename: "{app}\ContextAwareIQA.exe"
Name: "{autodesktop}\Context-Aware IQA"; Filename: "{app}\ContextAwareIQA.exe"

[Run]
Filename: "{app}\ContextAwareIQA.exe"; Description: "Launch Context-Aware IQA"; Flags: nowait postinstall skipifsilent
