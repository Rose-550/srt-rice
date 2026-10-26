<#
稻视界 · Git 首次初始化脚本（幂等：已初始化会直接退出）
================================================================
作用：在一个已经配好 .gitignore/.gitattributes 的项目里，安全地完成
      ① git init ② 仓库级身份 ③ 暂存并做安全检查 ④ 首次提交 ⑤ 打起点 tag
边界：只操作本地仓库，不联网、不推送；数据集/权重/日志/密钥由 .gitignore 排除。

用法（在项目根目录打开 PowerShell）：
    pwsh -File scripts\git_first_commit.ps1
    # 自定义身份：
    pwsh -File scripts\git_first_commit.ps1 -Name "罗浔" -Email "luoxun@local"
    # 不确认直接跑（你自己确认过的情况下）：
    pwsh -File scripts\git_first_commit.ps1 -Yes
#>
param(
    [string]$Name  = "罗浔",
    [string]$Email = "luoxun@local",
    [switch]$NoTag,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "    [OK] $t" -ForegroundColor Green }
function Warn($t) { Write-Host "    [!]  $t" -ForegroundColor Yellow }
function Die($t)  { Write-Host ""; Write-Host "[中止] $t" -ForegroundColor Red; exit 1 }

Step "0/6 检查环境"
$root = (Get-Location).Path
if (-not (Test-Path (Join-Path $root "rice_dp1_fluent.py"))) {
    Die "请先在项目根目录（含 rice_dp1_fluent.py）运行本脚本。当前目录：$root"
}
try { $gitVersion = (git --version) } catch { Die "没有找到 git，请先安装 Git for Windows（https://git-scm.com/download/win）" }
Ok $gitVersion

if (Test-Path (Join-Path $root ".git")) {
    Warn "检测到 .git，仓库已初始化；本脚本不重复执行。"
    Warn "如需查看状态：git status / git log --oneline"
    exit 0
}
if (-not $Yes) {
    Write-Host ""
    Warn "即将执行：git init -b main、仓库级身份配置、首次提交，并打一个起点 tag。"
    Warn "不会联网、不会推送；数据/权重/日志/密钥由 .gitignore 排除。"
    $answer = Read-Host "继续？输入 y 回车（其他任意键取消）"
    if ($answer -ne "y") { Write-Host "已取消，未做任何修改。"; exit 0 }
}

Step "1/6 初始化仓库"
git init -b main | Out-Host
if ($LASTEXITCODE -ne 0) {
    Warn "git init -b main 失败，尝试旧写法"
    git init | Out-Host
    git symbolic-ref HEAD refs/heads/main
}
Ok "仓库已建立：$root\.git"

Step "2/6 配置仓库级身份与显示"
git config --local user.name  $Name
git config --local user.email $Email
git config --local core.quotepath false
git config --local core.autocrlf false
Ok ("user.name  = " + (git config --local user.name))
Ok ("user.email = " + (git config --local user.email))

Step "3/6 暂存 + 安全检查"
git add -A
if ($LASTEXITCODE -ne 0) { Die "git add 失败，请查看上方报错" }

$staged = @(git diff --cached --name-only)
if ($staged.Count -eq 0) { Die "没有可提交的文件：检查 .gitignore 是否把所有东西都忽略了" }

$problems = New-Object System.Collections.Generic.List[string]
$totalBytes = 0
foreach ($f in $staged) {
    $fi = Get-Item -LiteralPath $f -ErrorAction SilentlyContinue
    if ($fi) { $totalBytes += $fi.Length }
    if ($f -like "*config.json")              { $problems.Add("含密钥的配置: $f") }
    if ($f -like "dataset_*/*")               { $problems.Add("数据集: $f") }
    if ($f -like "JPEGImages/*")              { $problems.Add("原始图池: $f") }
    if ($f -like "runs/*")                    { $problems.Add("训练产物: $f") }
    if ($f -like "*.pt" -or $f -like "*.onnx"){ $problems.Add("模型权重: $f") }
    if ($fi -and $fi.Length -gt 15MB)         { $problems.Add(("超大文件 " + [math]::Round($fi.Length/1MB,1) + " MB: " + $f)) }
}
Write-Host ("    暂存文件数: " + $staged.Count + "，合计 " + [math]::Round($totalBytes/1MB,1) + " MB")
if ($problems.Count -gt 0) {
    Write-Host "    发现不该进库的文件：" -ForegroundColor Red
    $problems | Select-Object -First 20 | ForEach-Object { Write-Host ("      - " + $_) -ForegroundColor Red }
    Write-Host "    处理建议：补 .gitignore 后执行  git reset  ，再重新运行本脚本。" -ForegroundColor Yellow
    Die "安全检查未通过，已停在提交之前（没有产生任何提交）"
}
Ok "安全检查通过：无密钥/数据集/权重/超大文件"

Step "4/6 首次提交"
git commit -m "chore: 初始化仓库（代码+文档基线，数据/权重/日志/密钥已排除）" | Out-Host
if ($LASTEXITCODE -ne 0) { Die "git commit 失败，请查看上方报错" }
Ok "首次提交完成"

Step "5/6 打起点 tag"
if (-not $NoTag) {
    $tag = "baseline-" + (Get-Date).ToString("yyyyMMdd")
    git tag -a $tag -m "git 管理起点：仓库初始化" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "已创建 tag: $tag" } else { Warn "tag 已存在或创建失败，跳过（不影响仓库）" }
}

Step "6/6 验证"
git log --oneline -1 | Out-Host
$tracked = @(git ls-files)
Ok ("当前被版本管理的文件数: " + $tracked.Count)
$dirty = @(git status --short)
if ($dirty.Count -eq 0) { Ok "工作区干净（nothing to commit）" } else { Warn ("工作区还有 " + $dirty.Count + " 项未提交（可能是被忽略的运行时文件，属正常）") }

Write-Host ""
Write-Host "完成。接下来：" -ForegroundColor Cyan
Write-Host "  1) 日常改动：git status -> git diff -> git add -A -> git commit -m '说明'"
Write-Host "  2) 回滚单个文件：git restore 文件名"
Write-Host "  3) 历史：git log --oneline"
Write-Host "  4) 详细教程见 docs/Git使用指南.md"
