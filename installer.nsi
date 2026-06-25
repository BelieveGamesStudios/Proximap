; installer.nsi
; NSIS Installer Script for Proximap

; =========================================================================
; GENERAL SETTINGS
; =========================================================================
!define PRODUCT_NAME "Proximap"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "ProximaXR Spatial Technologies"
!define PRODUCT_WEB_SITE "https://proximap.space"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\Proximap.exe"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

Name "${PRODUCT_NAME}"

; Target executable name
OutFile "Proximap_Setup.exe"

; Default installation folder
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"

; Registry key to store the installation path
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""

; Compress installer with solid LZMA compression for smallest setup file size
SetCompressor lzma

; Request administrator privileges for writing to Program Files
RequestExecutionLevel admin

; =========================================================================
; MODERN UI (MUI2) CONFIGURATION
; =========================================================================
!include "MUI2.nsh"

; Define custom icons (Must be .ico files)
; Adjust the paths below to point to your custom icon file.
!define MUI_ICON "app_icon.ico"
!define MUI_UNICON "app_icon.ico"

; Modern UI Pages
!insertmacro MUI_PAGE_WELCOME
; Uncomment the line below to show a license page. Just create a license.txt file.
; !insertmacro MUI_PAGE_LICENSE "license.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\Proximap.exe"
!insertmacro MUI_PAGE_FINISH

; Uninstaller Pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Language Settings
!insertmacro MUI_LANGUAGE "English"

; =========================================================================
; INSTALLATION SECTIONS
; =========================================================================
Section "Main Section" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite try

  ; Recursively copy the PyInstaller output directory
  ; This assumes you have run '.\package_app.ps1' or 'python -m PyInstaller' first
  File /r "dist\Proximap\*.*"

  ; Store installation folder in registry
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\Proximap.exe"
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Register the uninstaller in Windows Add/Remove Programs (Registry)
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\Proximap.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
SectionEnd

Section "Shortcuts" SEC02
  ; Create Start Menu Shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\Proximap.exe"
  CreateShortcut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninstall.exe"

  ; Create Desktop Shortcut
  CreateShortcut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\Proximap.exe"
SectionEnd

; =========================================================================
; UNINSTALLER SECTION
; =========================================================================
Section "Uninstall"
  ; Remove shortcuts
  Delete "$DESKTOP\${PRODUCT_NAME}.lnk"
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk"
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"

  ; Remove directories and files
  RMDir /r "$INSTDIR"

  ; Clean up registry keys
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"

  SetAutoClose true
SectionEnd
