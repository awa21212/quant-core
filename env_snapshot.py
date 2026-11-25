# -*- coding: utf-8 -*-
"""env_snapshot.py —— 把「这台电脑是什么情况」写成一页，供交接时检索。

用法：
  python env_snapshot.py            # 采集并写 rag/env/machine.md + machine.json
  python env_snapshot.py --quiet    # 只写文件，不打印

不做任何修改动作，只读。每条命令都有超时，采集失败只记「未知」不中断。
"""
from __future__ import annotations

import json, os, platform, shutil, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "env"
OUT.mkdir(exist_ok=True)


def list_wsl():
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                              r"Software\\Microsoft\\Windows\\CurrentVersion\\Lxss")
    except OSError:
        return "(无 WSL 注册表项)"
    out, i = [], 0
    while True:
        try:
            name = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        try:
            k = winreg.OpenKey(root, name)
            dist = winreg.QueryValueEx(k, "DistributionName")[0]
            ver = winreg.QueryValueEx(k, "Version")[0]
            st = winreg.QueryValueEx(k, "State")[0]
            out.append(f"{dist}  ver{ver}  state={st}")
        except OSError:
            continue
    return "\n".join(out) or "(空)"


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="ignore")
        out = (p.stdout or "") + (p.stderr or "")
        return out.strip()[:4000]
    except Exception as e:
        return f"<失败: {type(e).__name__}>"


def first_line(s):
    for ln in (s or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def collect():
    d = {}
    d["collected_at"] = datetime.now().isoformat(timespec="seconds")
    d["host"] = socket.gethostname()
    d["user"] = os.environ.get("USERNAME") or os.environ.get("USER")
    d["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    d["arch"] = platform.machine()

    # 硬件
    d["cpu"] = first_line(run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"'))
    d["cores"] = first_line(run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"'))
    d["ram_gb"] = first_line(run('powershell -NoProfile -Command '
                                 '"[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"'))
    d["gpu"] = first_line(run('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name"'))
    d["disks"] = run('powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | '
                     'Select-Object Name,@{n=\'GB\';e={[math]::Round($_.Used/1GB,1)}},'
                     '@{n=\'FreeGB\';e={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize"')

    # 工具链
    d["python"] = first_line(run("python --version"))
    d["python_exe"] = first_line(run('powershell -NoProfile -Command "(Get-Command python).Source"'))
    d["python_ver2"] = first_line(run('py -0p'))
    d["pip"] = first_line(run("python -m pip --version"))
    d["git"] = first_line(run("git --version"))
    d["node"] = first_line(run("node --version"))
    d["pnpm"] = first_line(run("pnpm.cmd --version"))
    d["gcc"] = first_line(run("gcc --version"))
    d["docker"] = first_line(run("docker --version"))
    d["docker_running"] = first_line(run("docker info --format '{{.ServerVersion}}'", timeout=60))
    d["images"] = run('docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}"')
    d["wsl"] = list_wsl()
    d["wsl_default"] = first_line(run('wsl -e sh -lc "echo $WSL_DISTRO_NAME $(uname -r)"'))
    d["ollama"] = first_line(run("ollama --version"))
    d["models"] = run("ollama list")

    # 代理与网络
    d["proxy"] = run('powershell -NoProfile -Command "Get-ItemProperty '
                     "'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings' "
                     '| Select-Object ProxyEnable,ProxyServer | Format-List"')
    d["ports"] = run('powershell -NoProfile -Command "Get-NetTCPConnection -State Listen '
                     '| Select-Object -ExpandProperty LocalPort -Unique | Sort-Object"')

    # 便携环境（如果有；路径由环境变量给，仓库里不写死任何机器路径）
    portable = os.environ.get("PORTABLE_ROOT")
    if portable and os.path.isdir(portable):
        d["portable"] = portable
        d["portable_items"] = run(f'powershell -NoProfile -Command "Get-ChildItem \'{portable}\' -Depth 0 | Select-Object Name"')

    # 关键目录：只列仓库自身，别把使用者的机器布局带进来
    dirs = ["quant_v5", "docs", "docker"]
    d["dirs"] = {x: os.path.isdir(str(BASE / x)) for x in dirs}

    # 定时任务
    d["tasks"] = run('powershell -NoProfile -Command "Get-ScheduledTask -TaskName \'Quant*\' '
                     '| Select-Object TaskName,State | Format-Table -AutoSize"')

    # 已装的关键 python 包
    d["py_packages"] = run("python -m pip list --format=freeze", timeout=40)
    return d


def render(d):
    L = []
    L.append("# 这台电脑是什么情况")
    L.append(f"\n> 采集时间：{d['collected_at']}　主机：{d['host']}　用户：{d['user']}")
    L.append("\n## 硬件")
    L.append(f"- CPU：{d['cpu']}（{d['cores']} 逻辑核）")
    L.append(f"- 内存：{d['ram_gb']} GB")
    L.append(f"- 显卡：{d['gpu']}")
    L.append("\n```\n" + d["disks"] + "\n```")
    L.append("## 系统")
    L.append(f"- {d['os']}　{d['arch']}")
    L.append("\n## 工具链")
    for k, label in [("python", "Python"), ("python_exe", "python 路径"), ("pip", "pip"),
                     ("git", "Git"), ("node", "Node"), ("pnpm", "pnpm"), ("gcc", "gcc"),
                     ("docker", "Docker"), ("docker_running", "Docker 引擎"),
                     ("ollama", "Ollama")]:
        if d.get(k):
            L.append(f"- {label}：{d[k]}")
    if d.get("wsl"):
        L.append("\n### WSL\n```\n" + d["wsl"] + "\n```")
    if d.get("wsl_default"):
        L.append(f"- 默认发行版：{d['wsl_default']}")
    if d.get("portable"):
        L.append(f"\n### 便携环境\n- 位置：{d['portable']}（VS Code + Python + Git + MinGW）")
    L.append("\n## 代理与端口")
    L.append("```\n" + (d.get("proxy") or "") + "\n```")
    L.append("> 系统代理 127.0.0.1:7892 时死时活；GitHub 直连常被 reset，拉大文件用 ghfast.top 前缀 + 分片校验。")
    L.append("\n## 关键目录")
    for k, v in (d.get("dirs") or {}).items():
        L.append(f"- `{k}`：{'存在' if v else '不存在'}")
    if d.get("tasks"):
        L.append("\n## 定时任务\n```\n" + d["tasks"] + "\n```")
    if d.get("images"):
        L.append("\n## Docker 镜像\n```\n" + d["images"] + "\n```")
    if d.get("models"):
        L.append("\n## 本地模型\n```\n" + d["models"] + "\n```")
    pk = (d.get("py_packages") or "")
    keys = ("pandas", "numpy", "akshare", "torch", "paddle", "matplotlib", "requests",
            "python-docx", "onnx", "pybind11")
    picked = [ln for ln in pk.splitlines() if ln.split("==")[0].lower() in keys]
    if picked:
        L.append("\n## 关键 Python 包\n```\n" + "\n".join(picked) + "\n```")
    return "\n".join(L) + "\n"


def main():
    quiet = "--quiet" in sys.argv
    t0 = time.time()
    d = collect()
    md = render(d)
    (OUT / "machine.md").write_text(md, encoding="utf-8")
    (OUT / "machine.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    if not quiet:
        print(md)
    print(f"[ok] {OUT / 'machine.md'}  ({len(md)} 字符, {time.time()-t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

