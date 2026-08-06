; Instalador do Scene Finder (Inno Setup)
; Instala por usuario (sem UAC) e substitui a versao anterior no mesmo AppId.

#define AppName "Scene Finder"
#define AppExe "SceneFinder.exe"
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8E3A6F41-2C7D-4B95-9E10-5D4C2A7B6E33}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=RaymundoJMSN
AppPublisherURL=https://github.com/RaymundoJMSN/scene-finder
DefaultDirName={autopf}\SceneFinder
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir=dist
OutputBaseFilename=SceneFinder-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
; fecha o app antes de substituir os arquivos (updates)
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na Área de Trabalho"; GroupDescription: "Atalhos:"

[Files]
Source: "dist\SceneFinder\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Abrir o {#AppName}"; Flags: nowait postinstall skipifsilent

; o indice do usuario fica em LOCALAPPDATA e sobrevive a updates;
; so e removido se ele pedir na desinstalacao
[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Dados: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    Dados := ExpandConstant('{localappdata}\SceneFinder');
    if DirExists(Dados) then
      if MsgBox('Remover também o índice e as miniaturas? (' + Dados + ')',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(Dados, True, True, True);
  end;
end;
