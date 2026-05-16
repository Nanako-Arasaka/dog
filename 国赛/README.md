# 国赛代码框架（绝影 Lite2 / UDP 协议）

这是一个可部署到机器狗控制主机的**国赛三阶段框架**：
1. 避障通过
2. 巡检识别与语音播报
3. 按巡检异常结果执行红色长条抓取投放

## 目录结构

```text
国赛
├─ config
│  └─ robot_config.json
├─ scripts
│  ├─ deploy.ps1
│  └─ run.ps1
└─ src
   ├─ dog_sdk
   │  ├─ commands.py
   │  ├─ protocol.py
   │  ├─ telemetry.py
   │  └─ transport.py
   ├─ mission
   │  ├─ base.py
   │  ├─ models.py
   │  ├─ perception.py
   │  └─ national_stage.py
   ├─ runtime
   │  ├─ controller.py
   │  └─ speaker.py
   └─ main.py
```

## 快速运行（本机联调）

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\run.ps1
```

## 部署到机器狗上位机（示例）

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\deploy.ps1 -TargetUser ubuntu -TargetHost 192.168.1.103 -TargetDir /home/ubuntu/national_stage
```

> 你需要把 `config\robot_config.json` 里的 IP/端口改成现场实际配置。

## 已实现的任务逻辑（骨架）

- 避障阶段：持续前进 + 超时保护，等待感知侧给出“已通过障碍区”
- 巡检阶段：读取 A/B/C/D 仪表状态，按要求播报“偏低/偏高/正常 + 异常/正常”
- 抓取阶段：根据异常区域队列执行红色长条投放，掉落累计到 3 次即判失败

## 你需要替换的接口

- `src\mission\perception.py`
  - `obstacle_cleared()`
  - `poll_inspection()`
  - `execute_pickup_for_zone()`

当前默认使用 `config\scenario_mock.json` 做仿真，便于你不连真机先联调任务状态机。
