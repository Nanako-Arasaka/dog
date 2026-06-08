# 国赛代码架构文档

## 概述

本项目是绝影 Lite2 四足机器人参加国赛的完整软件框架。采用**分层架构 + 依赖注入**模式，所有硬件与算法模块通过 ABC 接口隔离，支持仿真/实机无缝切换。

## 分层架构

```
┌──────────────────────────────────────────────┐
│  main.py  (入口)                             │
│  app/     (配置 + DI 容器)                    │
├──────────────────────────────────────────────┤
│  mission/ (任务状态机)                        │
│    ├─ base.py          MissionBase ABC       │
│    ├─ national_stage.py  国赛 10 阶段状态机    │
│    └─ phase.py         PhaseHandler ABC      │
├──────────┬──────────────┬────────────────────┤
│perception│  navigation  │  runtime/           │
│gateway   │  gateway     │  controller.py      │
├──────────┼──────────────┼────────────────────┤
│camera    │  camera      │  dog_sdk/           │
│arm       │  dog_sdk     │  (UDP 协议栈)       │
│detector/*│              │                    │
├──────────┴──────────────┴────────────────────┤
│  core/  (领域类型 + 异常)                      │
│  utils/ (工具函数)                            │
└──────────────────────────────────────────────┘
```

## 模块职责

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `core/` | 领域类型、枚举、异常 | `Zone`, `MeterStatus`, `MissionPhase` |
| `app/` | 配置加载、DI 容器 | `AppConfig`, `AppContainer` |
| `hardware/camera/` | 相机抽象 | `CameraGateway` |
| `hardware/arm/` | 机械臂抽象 + 运动学 | `ArmGateway`, `KinematicSolver` |
| `hardware/speaker/` | 语音播报抽象 | `SpeakerGateway` |
| `dog_sdk/` | 绝影 UDP 协议栈 | `DogController`, `UdpTransport` |
| `perception/` | 感知编排 + 检测器基类 | `PerceptionGateway`, `BaseDetector` |
| `navigation/` | 定位 + 路径规划 | `NavigationGateway` |
| `mission/` | 任务状态机 | `NationalStageMission`, `PhaseHandler` |
| `utils/` | 计时、坐标变换 | `RateLimiter`, `pixel_to_camera_3d` |

## 任务流程（10 阶段状态机）

```
INIT → OBSTACLE_APPROACH → OBSTACLE_DETECT → OBSTACLE_CROSS
  → INSPECTION_NAV → INSPECTION_SCAN → INSPECTION_READ
  → PICKUP_PLAN → PICKUP_NAV → PICKUP_GRAB → PICKUP_TRANSPORT → PICKUP_PLACE
  → DONE / FAILED
```

## 配置说明

编辑 `config/robot_config.json`：

- `robot`: 机器狗 IP/端口
- `timing`: 心跳/主循环频率
- `camera.driver`: `"mock"` (仿真) 或 `"realsense"` (实机)
- `arm.driver`: `"mock"` 或具体型号驱动名
- `speaker.engine`: `"mock"` / `"espeak"`
- `mission`: 任务超时/重试参数
- `perception`: 模型目录 / 置信度阈值
- `scenario_file`: 仿真场景 JSON 路径

## 运行方式

```powershell
# 仿真联调（使用 scenario_mock.json）
.\scripts\run.ps1

# 部署到机器狗
.\scripts\deploy.ps1 -TargetUser ubuntu -TargetHost 192.168.1.103 -TargetDir /home/ubuntu/national_stage
```

## 扩展指南

1. **新增相机驱动**: 实现 `CameraGateway`，在 `_create_camera()` 中注册。
2. **新增机械臂驱动**: 实现 `ArmGateway` + 可选 `KinematicSolver`，在 `_create_arm()` 中注册。
3. **新增检测器**: 继承 `BaseDetector[T]`，在 `PerceptionGateway` 实现中编排。
4. **新增任务阶段**: 创建 `PhaseHandler` 子类，注册到状态机的 `_phase_handlers` 字典。
5. **写测试**: 在 `tests/` 下新增测试文件，使用 `conftest.py` 中的 mock fixtures。
