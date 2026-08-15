' digimonUp 매크로 - 콘솔 창 없이 GUI 만 띄우는 실행기
'
' run.bat 은 배치 파일이라 어쩔 수 없이 검은 cmd 창이 같이 뜬다.
' 이 파일을 더블클릭하면 pythonw.exe(콘솔 없는 파이썬)로 실행해서
' GUI 창 하나만 뜬다. cmd 창은 잠깐도 보이지 않는다.

Option Explicit

Dim fso, shell, baseDir, launcher, pythonw, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' 이 스크립트는 scripts/ 안에 있고 실제 프로젝트는 그 위 폴더다.
baseDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
launcher = fso.BuildPath(baseDir, "launcher.py")

If Not fso.FileExists(launcher) Then
    MsgBox "launcher.py 를 찾을 수 없습니다:" & vbCrLf & launcher, 16, "digimonUp"
    WScript.Quit 1
End If

' EXE 가 빌드돼 있으면 그걸 먼저 쓴다 (이미 콘솔 없는 창 모드로 빌드됨).
Dim exePath
exePath = fso.BuildPath(baseDir, "dist\digimonUp.exe")
If fso.FileExists(exePath) Then
    shell.CurrentDirectory = fso.GetParentFolderName(exePath)
    shell.Run """" & exePath & """", 1, False
    WScript.Quit 0
End If

pythonw = FindPythonw()
If pythonw = "" Then
    MsgBox "pythonw.exe 를 찾을 수 없습니다." & vbCrLf & _
           "파이썬이 설치돼 있는지 확인해주세요.", 16, "digimonUp"
    WScript.Quit 1
End If

shell.CurrentDirectory = baseDir
' 세 번째 인자 False = 끝날 때까지 기다리지 않음, 두 번째 0 = 창 숨김
shell.Run """" & pythonw & """ """ & launcher & """", 0, False

' ----------------------------------------------------------------------
Function FindPythonw()
    Dim candidates, c, out
    FindPythonw = ""

    ' 1) py 런처가 알려주는 경로 옆의 pythonw
    On Error Resume Next
    Dim exec, pyPath
    Set exec = shell.Exec("cmd /c where python")
    If Err.Number = 0 Then
        pyPath = Trim(Split(exec.StdOut.ReadAll(), vbCrLf)(0))
        If pyPath <> "" And fso.FileExists(pyPath) Then
            out = fso.BuildPath(fso.GetParentFolderName(pyPath), "pythonw.exe")
            If fso.FileExists(out) Then
                FindPythonw = out
                Exit Function
            End If
        End If
    End If
    Err.Clear
    On Error GoTo 0

    ' 2) 흔한 설치 위치
    candidates = Array( _
        shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Programs\Python"), _
        "C:\Python312", "C:\Python311", "C:\Program Files\Python312")
    For Each c In candidates
        out = SearchPythonw(c)
        If out <> "" Then
            FindPythonw = out
            Exit Function
        End If
    Next
End Function

Function SearchPythonw(root)
    Dim folder, sub_, direct
    SearchPythonw = ""
    If Not fso.FolderExists(root) Then Exit Function

    direct = fso.BuildPath(root, "pythonw.exe")
    If fso.FileExists(direct) Then
        SearchPythonw = direct
        Exit Function
    End If

    Set folder = fso.GetFolder(root)
    For Each sub_ In folder.SubFolders
        direct = fso.BuildPath(sub_.Path, "pythonw.exe")
        If fso.FileExists(direct) Then
            SearchPythonw = direct
            Exit Function
        End If
    Next
End Function
