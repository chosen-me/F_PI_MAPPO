"""
CACC实时仿真环境（完整修正版）
✅ 9维状态全部从SUMO实时获取
✅ 禁用SUMO默认控制，纯加速度控制
✅ 完整的多目标奖励函数
✅ 完善的错误处理和仿真稳定性保障
"""
import traci
import numpy as np
import time
import random
from datetime import datetime
import os
import sys


class CACCRealTimeEnv:
    def __init__(self, config=None):
        """初始化仿真环境"""
        # 配置合并
        default_config = self.get_default_config()
        self.config = {**default_config, **(config or {})}

        # 新增：场景类型记录
        self.scenario_type = self.config.get('scenario_type', 'straight')
        # 根据场景类型选择SUMO配置文件
        # 修正第25-33行
        if self.scenario_type == 'straight':
            self.config['sumo_config'] = 'straight.sumocfg'
        elif self.scenario_type == 'curve_left':
            self.config['sumo_config'] = 'left.sumocfg'
        elif self.scenario_type == 'curve_right':
            self.config['sumo_config'] = 'right.sumocfg'
        else:
            self.config['sumo_config'] = 'straight.sumocfg'

        # 车辆编队配置
        self.leader_id = 'leader'
        self.follower_ids = ['follower1', 'follower2', 'follower3']
        self.all_vehicles = [self.leader_id] + self.follower_ids

        # 车辆物理参数
        self.vehicle_length = 4.5  # 车辆长度（米）
        self.desired_spacing = 15.0  # 期望车间距（米）
        self.safe_spacing = 5.0  # 最小安全间距（米）
        self.max_acceleration = 3.0  # 最大加速度（m/s²）

        # 仿真参数
        self.simulation_step = self.config['step_length']
        self.max_simulation_time = self.config['max_steps_per_episode'] * self.simulation_step
        self.current_time = 0.0
        self.step_count = 0

        # 领头车扰动配置
        self.disturbance_enabled = True
        self.disturbance_types = ['cruise', 'acceleration', 'deceleration', 'stop_and_go']
        self.current_disturbance = 'cruise'
        self.disturbance_duration = 0
        self.disturbance_max_duration = 50  # 5秒切换一次扰动

        # 扰动参数
        self.cruise_speed = 20.0  # 巡航速度（m/s）
        self.acceleration_target = 25.0  # 加速目标速度
        self.deceleration_target = 15.0  # 减速目标速度

        # 环境状态
        self.is_connected = False
        self.collision_detected = False
        self.continuous_collision_count = 0

        # 数据记录
        self.episode_data = self._init_episode_data()

        # 历史数据缓存（用于计算jerk）
        self.prev_accelerations = [0.0, 0.0, 0.0]

        print(f"🎯 CACC实时环境初始化完成")
        print(f"  车辆编队：{self.all_vehicles}")
        print(f"  状态维度：9维（全部实时获取）")
        print(f"  动作范围：[-{self.max_acceleration}, {self.max_acceleration}] m/s²")
        print(f"  仿真步长：{self.simulation_step}s")

    def get_default_config(self):
        """默认配置"""
        return {
            'sumo_config': 'straight.sumocfg',
            'gui': False,
            'step_length': 0.1,
            'collision_check': True,
            'max_steps_per_episode': 500,
            'reward_weights': {
                'spacing': 1.0,
                'speed': 1.0,
                'acceleration': 0.1,
                'jerk': 0.1,
                'collision': 10.0
            }
        }

    def _init_episode_data(self):
        """初始化回合数据记录"""
        return {
            'time': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'leader_speeds': [],
            'spacings': [],
            'collisions': []
        }

    def check_environment(self):
        """检查SUMO配置文件是否齐全"""
        required_files = [
            self.config['sumo_config'],
            'straight.net.xml',
            'straight.rou.xml'
        ]

        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ 缺失必要文件：{file}")
                return False

        print("✅ 环境配置检查通过")
        return True

    def connect_to_sumo(self):
        """连接SUMO仿真（带重连机制）"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                print(f"连接SUMO（第{retry + 1}次尝试）...")
                # 构建启动命令
                sumo_binary = 'sumo-gui' if self.config['gui'] else 'sumo'
                sumo_cmd = [
                    sumo_binary,
                    '-c', self.config['sumo_config'],
                    '--step-length', str(self.simulation_step),
                    '--collision.action', 'warn',
                    '--no-warnings', 'true',
                    '--time-to-teleport', '-1',
                    '--waiting-time-memory', '1000',
                    '--start'
                ]

                # 启动并连接
                traci.start(sumo_cmd)
                self.is_connected = True
                time.sleep(1)  # 等待连接稳定
                print("✅ SUMO连接成功")
                return True

            except Exception as e:
                print(f"❌ 第{retry + 1}次连接失败：{e}")
                if retry < max_retries - 1:
                    time.sleep(2)
                    continue

        return False

    def setup_vehicles(self):
        """设置车辆控制模式（禁用SUMO默认控制）"""
        try:
            print("配置车辆控制模式...")
            # 等待所有车辆加载完成
            max_wait_steps = 50  # 最多等待5秒
            for _ in range(max_wait_steps):
                vehicle_list = traci.vehicle.getIDList()
                if all(veh_id in vehicle_list for veh_id in self.all_vehicles):
                    break
                traci.simulationStep()
                time.sleep(0.01)

            # 检查车辆是否全部加载
            vehicle_list = traci.vehicle.getIDList()
            missing_vehicles = [v for v in self.all_vehicles if v not in vehicle_list]
            if missing_vehicles:
                print(f"❌ 车辆缺失：{missing_vehicles}")
                return False

            # 配置所有车辆（禁用默认控制）
            for veh_id in self.all_vehicles:
                # 完全禁用SUMO默认速度控制
                traci.vehicle.setSpeedMode(veh_id, 0)
                # 禁止变道
                traci.vehicle.setLaneChangeMode(veh_id, 0)
                # 设置初始速度和加速度
                traci.vehicle.setSpeed(veh_id, 2.0)
                traci.vehicle.setAcceleration(veh_id, 0.0, 0.1)
                # 设置最大速度（确保加速度控制生效）
                traci.vehicle.setMaxSpeed(veh_id, 100.0)

            print("✅ 车辆配置完成（已禁用SUMO默认控制）")
            return True

        except Exception as e:
            print(f"❌ 车辆配置失败：{e}")
            return False

    def get_state(self, vehicle_id):
        """
        获取单个车辆的9维状态向量（全部从SUMO实时获取）
        状态构成：[自车速度, 自车加速度, 与前车间距, 与前车相对速度, 
                  与领头车间距, 与领头车速度差, 道路曲率, 前车加速度, 领头车加速度]
        """
        try:
            # 1. 自车速度（m/s）
            speed = max(0.0, traci.vehicle.getSpeed(vehicle_id))

            # 2. 自车加速度（m/s²）
            acceleration = traci.vehicle.getAcceleration(vehicle_id)

            # 确定前车和领头车
            if vehicle_id == 'follower1':
                front_vehicle = 'leader'
            elif vehicle_id == 'follower2':
                front_vehicle = 'follower1'
            elif vehicle_id == 'follower3':
                front_vehicle = 'follower2'
            else:  # leader无前车
                front_vehicle = None

            # 3. 与前车间距（m）；4. 与前车相对速度（m/s）；8. 前车加速度（m/s²）
            spacing = self.desired_spacing
            relative_speed = 0.0
            front_acceleration = 0.0

            if front_vehicle and front_vehicle in traci.vehicle.getIDList():
                # 获取前车和自车的车道位置
                front_lane_pos = traci.vehicle.getLanePosition(front_vehicle)
                ego_lane_pos = traci.vehicle.getLanePosition(vehicle_id)
                # 计算实际间距（减去车辆长度）
                spacing = max(0.1, front_lane_pos - ego_lane_pos - self.vehicle_length)
                # 相对速度
                front_speed = traci.vehicle.getSpeed(front_vehicle)
                relative_speed = front_speed - speed
                # 前车加速度
                front_acceleration = traci.vehicle.getAcceleration(front_vehicle)

            # 5. 与领头车间距（m）；6. 与领头车速度差（m/s）；9. 领头车加速度（m/s²）
            distance_to_leader = 0.0
            speed_diff_leader = 0.0
            leader_acceleration = 0.0

            if self.leader_id in traci.vehicle.getIDList():
                leader_lane_pos = traci.vehicle.getLanePosition(self.leader_id)
                ego_lane_pos = traci.vehicle.getLanePosition(vehicle_id)
                distance_to_leader = max(0.1, leader_lane_pos - ego_lane_pos - self.vehicle_length)
                leader_speed = traci.vehicle.getSpeed(self.leader_id)
                speed_diff_leader = leader_speed - speed
                leader_acceleration = traci.vehicle.getAcceleration(self.leader_id)

            # 7. 道路曲率（直道为0，弯道场景可扩展）
            road_curvature = 0.0

            # 构建9维状态向量
            state = np.array([
                speed,  # 1. 自车速度
                acceleration,  # 2. 自车加速度
                spacing,  # 3. 与前车间距
                relative_speed,  # 4. 与前车相对速度
                distance_to_leader,  # 5. 与领头车间距
                speed_diff_leader,  # 6. 与领头车速度差
                road_curvature,  # 7. 道路曲率
                front_acceleration,  # 8. 前车加速度
                leader_acceleration  # 9. 领头车加速度
            ], dtype=np.float32)

            return state

        except Exception as e:
            print(f"⚠️ 获取车辆{vehicle_id}状态失败：{e}")
            # 返回安全默认状态
            return np.array([5.0, 0.0, self.desired_spacing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_all_states(self):
        """获取所有跟随车辆的状态"""
        return [self.get_state(follower_id) for follower_id in self.follower_ids]

    def apply_leader_disturbance(self):
        """应用领头车扰动（模拟真实交通场景）"""
        if not self.disturbance_enabled:
            return

        self.disturbance_duration += 1
        # 切换扰动模式
        if self.disturbance_duration >= self.disturbance_max_duration:
            self.current_disturbance = random.choice(self.disturbance_types)
            self.disturbance_duration = 0
            print(f"🔄 切换领头车扰动模式：{self.current_disturbance}")

        try:
            current_speed = traci.vehicle.getSpeed(self.leader_id)

            # 根据扰动类型计算加速度
            if self.current_disturbance == 'cruise':
                # 巡航模式：维持恒定速度
                target_speed = self.cruise_speed
                if current_speed < target_speed - 0.5:
                    acceleration = 1.0
                elif current_speed > target_speed + 0.5:
                    acceleration = -1.0
                else:
                    acceleration = 0.0

            elif self.current_disturbance == 'acceleration':
                # 加速模式
                target_speed = self.acceleration_target
                acceleration = 2.0 if current_speed < target_speed else 0.0

            elif self.current_disturbance == 'deceleration':
                # 减速模式
                target_speed = self.deceleration_target
                acceleration = -2.0 if current_speed > target_speed else 0.0

            elif self.current_disturbance == 'stop_and_go':
                # 启停模式
                if self.disturbance_duration < 25:  # 前2.5秒减速停车
                    acceleration = -3.0 if current_speed > 0.5 else 0.0
                else:  # 后2.5秒加速
                    acceleration = 2.0 if current_speed < self.cruise_speed else 0.0

            else:
                acceleration = 0.0

            # 应用加速度控制
            traci.vehicle.setAcceleration(self.leader_id, acceleration, self.simulation_step)

        except Exception as e:
            print(f"⚠️ 应用领头车扰动失败：{e}")

    def apply_follower_actions(self, actions):
        """
        应用跟随车辆动作（纯加速度控制）
        :param actions: 动作列表（归一化到[-1,1]）
        :return: 实际应用的加速度
        """
        applied_actions = []

        for i, (follower_id, action) in enumerate(zip(self.follower_ids, actions)):
            try:
                # 解析动作（确保是标量）
                if isinstance(action, (list, np.ndarray)):
                    normalized_action = float(action[0]) if len(action) > 0 else 0.0
                else:
                    normalized_action = float(action)

                # 裁剪到[-1,1]并映射到实际加速度
                normalized_action = np.clip(normalized_action, -1.0, 1.0)
                actual_acceleration = normalized_action * self.max_acceleration

                # 关键：禁用SUMO默认控制（每次都设置确保生效）
                traci.vehicle.setSpeedMode(follower_id, 0)
                traci.vehicle.setLaneChangeMode(follower_id, 0)

                # 应用加速度控制（持续时间=仿真步长）
                traci.vehicle.setAcceleration(follower_id, actual_acceleration, self.simulation_step)
                # 设置最大速度确保加速度控制生效
                traci.vehicle.setMaxSpeed(follower_id, 100.0)

                applied_actions.append(actual_acceleration)

                # 调试信息（每100步打印一次）
                if self.step_count % 100 == 0:
                    current_speed = traci.vehicle.getSpeed(follower_id)
                    print(f"🚗 车辆{follower_id}：速度={current_speed:.2f}m/s，加速度={actual_acceleration:.2f}m/s²")

            except Exception as e:
                print(f"⚠️ 设置车辆{follower_id}动作失败：{e}")
                applied_actions.append(0.0)

        return applied_actions

    def compute_reward(self, states, actions):
        """
        多目标奖励函数（安全>稳定>效率>舒适）
        :param states: 所有跟随车辆的状态
        :param actions: 所有跟随车辆的动作
        :return: 每个车辆的奖励
        """
        weights = self.config['reward_weights']
        rewards = []

        # 获取领头车速度
        try:
            leader_speed = traci.vehicle.getSpeed(self.leader_id)
        except:
            leader_speed = self.cruise_speed

        for i, (state, action) in enumerate(zip(states, actions)):
            # 状态解析
            speed = state[0]
            acceleration = state[1]
            spacing = state[2]
            speed_diff_leader = state[5]

            if spacing < 3.0:  # 危险区间：轻罚（原-100→-30）
                spacing_reward = -30.0 * weights['collision']
            elif spacing < self.safe_spacing:  # 警告区间：中罚（不变）
                spacing_reward = -20.0 * (self.safe_spacing - spacing) / self.safe_spacing * weights['spacing']
            else:  # 安全区间：轻罚+正奖励（鼓励维持安全）
                spacing_error = abs(spacing - self.desired_spacing)
                spacing_reward = (-spacing_error / self.desired_spacing * (weights['spacing'] * 0.5)) + 5.0  # 加5点正奖励

            # 2. 效率性奖励（速度惩罚）
            speed_error = abs(speed - leader_speed)
            speed_reward = -speed_error / 5.0 * weights['speed']

            # 3. 舒适性奖励（加速度惩罚）
            accel_penalty = (acceleration / self.max_acceleration) ** 2  # 平方惩罚
            accel_reward = -accel_penalty * weights['acceleration']

            # 4. 舒适性奖励（加加速度jerk惩罚）
            jerk = abs(acceleration - self.prev_accelerations[i]) / self.simulation_step
            jerk_penalty = (jerk / 10.0) ** 2
            jerk_reward = -min(jerk_penalty, 1.0) * weights['jerk']

            # 5. 稳定性奖励（速度差惩罚）
            stability_reward = -abs(speed_diff_leader) / 5.0 * 0.5

            # 6. 动作平滑性惩罚
            action_value = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            action_reward = -abs(action_value) / 2.0 * 0.1

            # 总奖励（所有惩罚项之和）
            total_reward = (
                    spacing_reward +
                    speed_reward +
                    accel_reward +
                    jerk_reward +
                    stability_reward +
                    action_reward
            )

            rewards.append(total_reward)

        # 更新历史加速度（用于下一次jerk计算）
        self.prev_accelerations = [state[1] for state in states]

        return rewards

    def check_collisions(self):
        """检查碰撞"""
        try:
            collisions = traci.simulation.getCollisions()
            if collisions:
                self.collision_detected = True
                print(f"🚨 检测到碰撞：{collisions}")
                return True
        except:
            pass
        return False

    def check_termination(self):
        # 1. 达到最大仿真时间 → 正常终止
        if self.current_time >= self.max_simulation_time:
            return True

        # 2. 碰撞处理：不直接终止，记录连续碰撞次数
        if self.config['collision_check'] and self.check_collisions():
            self.continuous_collision_count += 1
            if self.continuous_collision_count >= 5:  # 连续5步碰撞才终止
                return True
        else:
            self.continuous_collision_count = 0  # 重置连续碰撞计数

        # 3. 车辆丢失 → 终止
        try:
            vehicle_list = traci.vehicle.getIDList()
            for veh_id in self.all_vehicles:
                if veh_id not in vehicle_list:
                    return True
        except:
            pass

        return False

    def reset(self):
        """重置环境"""
        print("\n" + "=" * 50)
        print("🔄 重置仿真环境")
        print("=" * 50)

        # 关闭现有连接
        if self.is_connected:
            try:
                traci.close()
            except:
                pass

        # 重置状态
        self.current_time = 0.0
        self.step_count = 0
        self.collision_detected = False
        self.disturbance_duration = 0
        self.current_disturbance = 'cruise'
        self.episode_data = self._init_episode_data()
        self.prev_accelerations = [0.0, 0.0, 0.0]

        # 检查环境配置
        if not self.check_environment():
            raise RuntimeError("环境配置不完整，无法启动仿真")

        # 连接SUMO
        if not self.connect_to_sumo():
            raise RuntimeError("SUMO连接失败")

        # 配置车辆
        if not self.setup_vehicles():
            raise RuntimeError("车辆配置失败")

        # 获取初始状态
        initial_states = self.get_all_states()

        # 验证状态有效性
        self._validate_states(initial_states)

        # 记录初始数据
        self.episode_data['time'].append(self.current_time)
        self.episode_data['states'].append(initial_states)

        try:
            leader_speed = traci.vehicle.getSpeed(self.leader_id)
            self.episode_data['leader_speeds'].append(leader_speed)
        except:
            self.episode_data['leader_speeds'].append(self.cruise_speed)

        print(f"✅ 环境重置完成")
        print(f"  初始领头车速度：{self.episode_data['leader_speeds'][-1]:.2f}m/s")

        return initial_states

    def _validate_states(self, states):
        """验证状态数据的有效性"""
        print("\n📊 状态有效性验证")
        for i, (follower_id, state) in enumerate(zip(self.follower_ids, states)):
            print(f"  车辆{follower_id}的9维状态：")
            print(f"    1. 自车速度：{state[0]:.2f}m/s（有效范围：0-30）")
            print(f"    2. 自车加速度：{state[1]:.2f}m/s²（有效范围：-3-3）")
            print(f"    3. 与前车间距：{state[2]:.2f}m（有效范围：5-50）")
            print(f"    4. 与前车相对速度：{state[3]:.2f}m/s")
            print(f"    5. 与领头车间距：{state[4]:.2f}m")
            print(f"    6. 与领头车速度差：{state[5]:.2f}m/s")
            print(f"    7. 道路曲率：{state[6]:.2f}")
            print(f"    8. 前车加速度：{state[7]:.2f}m/s²")
            print(f"    9. 领头车加速度：{state[8]:.2f}m/s²")

            # 简单有效性检查
            assert 0 <= state[0] <= 30, f"速度超出合理范围：{state[0]}"
            assert -5 <= state[1] <= 5, f"加速度超出合理范围：{state[1]}"
            assert state[2] >= 0.1, f"间距为负：{state[2]}"

        print("✅ 状态验证通过（全部实时获取）")

    def step(self, actions):
        """
        执行一步仿真
        :param actions: 所有跟随车辆的动作
        :return: next_states, rewards, done, info
        """
        try:
            # 应用领头车扰动
            self.apply_leader_disturbance()

            # 应用跟随车辆动作
            applied_actions = self.apply_follower_actions(actions)

            # 执行仿真步进
            traci.simulationStep()

            # 更新时间
            self.current_time = traci.simulation.getTime()
            self.step_count += 1

            # 获取下一状态
            next_states = self.get_all_states()

            # 计算奖励
            rewards = self.compute_reward(next_states, actions)

            # 检查终止条件
            done = self.check_termination()

            # 记录数据
            self._record_episode_data(next_states, applied_actions, rewards)

            # 构建信息字典
            info = self._build_info_dict(applied_actions, rewards, next_states)

            # 进度打印（每50步）
            if self.step_count % 50 == 0:
                avg_reward = np.mean(rewards)
                avg_spacing = np.mean([s[2] for s in next_states])
                print(f"📈 步{self.step_count:4d}：时间={self.current_time:6.1f}s，"
                      f"平均奖励={avg_reward:7.3f}，平均间距={avg_spacing:5.2f}m，"
                      f"领头车速={info['leader_speed']:5.2f}m/s")

            return next_states, rewards, done, info

        except Exception as e:
            print(f"❌ 仿真步进失败：{e}")
            import traceback
            traceback.print_exc()

            # 返回安全的错误状态
            error_info = {
                'time': self.current_time,
                'step': self.step_count,
                'disturbance': self.current_disturbance,
                'collision': self.collision_detected,
                'leader_speed': self.cruise_speed,
                'spacings': [0.0] * 3,
                'rewards': [-100.0] * 3,
                'applied_actions': [0.0] * 3,
                'error': str(e)
            }

            return self.get_all_states(), [-100.0] * 3, True, error_info

    def _record_episode_data(self, next_states, applied_actions, rewards):
        """记录回合数据"""
        self.episode_data['time'].append(self.current_time)
        self.episode_data['states'].append(next_states)
        self.episode_data['actions'].append(applied_actions)
        self.episode_data['rewards'].append(rewards)

        # 记录领头车速度
        try:
            leader_speed = traci.vehicle.getSpeed(self.leader_id)
            self.episode_data['leader_speeds'].append(leader_speed)
        except:
            self.episode_data['leader_speeds'].append(self.cruise_speed)

        # 记录间距
        spacings = [state[2] for state in next_states]
        self.episode_data['spacings'].append(spacings)

        # 记录碰撞
        self.episode_data['collisions'].append(self.collision_detected)

    def _build_info_dict(self, applied_actions, rewards, next_states):
        """构建信息字典"""
        spacings = [state[2] for state in next_states]
        return {
            'time': self.current_time,
            'step': self.step_count,
            'disturbance': self.current_disturbance,
            'collision': self.collision_detected,
            'leader_speed': self.episode_data['leader_speeds'][-1],
            'spacings': spacings,
            'rewards': rewards,
            'applied_actions': applied_actions,
            'ttc': self._compute_ttc(next_states)
        }

    def _compute_ttc(self, states):
        """计算时间碰撞余量（TTC）"""
        ttc_values = []
        for state in states:
            spacing = state[2]
            relative_speed = state[3]  # 前车速度 - 自车速度

            if relative_speed < 0:  # 正在接近前车
                ttc = max(0.1, spacing / abs(relative_speed))
            else:
                ttc = float('inf')

            ttc_values.append(ttc)

        return ttc_values

    def get_performance_metrics(self):
        """计算性能指标（安全性、效率性、舒适性、稳定性）"""
        if len(self.episode_data['states']) == 0:
            return {}

        try:
            states_array = np.array(self.episode_data['states'])
            actions_array = np.array(self.episode_data['actions'])
            rewards_array = np.array(self.episode_data['rewards'])
            leader_speeds = np.array(self.episode_data['leader_speeds'])
            spacings_array = np.array(self.episode_data['spacings'])
            collisions = np.array(self.episode_data['collisions'])

            # 安全性指标
            min_spacings = np.min(spacings_array, axis=0)
            ttc_values = np.array([self._compute_ttc(states) for states in self.episode_data['states']])
            ttc_below_3 = np.sum(ttc_values < 3.0, axis=0) / len(ttc_values) * 100  # 百分比
            collision_count = np.sum(collisions)

            # 效率性指标
            avg_speeds = np.mean(states_array[:, :, 0], axis=0)
            speed_std = np.std(states_array[:, :, 0], axis=0)
            avg_leader_speed = np.mean(leader_speeds)
            speed_following_error = np.mean(np.abs(states_array[:, :, 0] - leader_speeds[:, None]), axis=0)

            # 舒适性指标
            avg_accel_abs = np.mean(np.abs(states_array[:, :, 1]), axis=0)
            accel_rms = np.sqrt(np.mean(states_array[:, :, 1] ** 2, axis=0))

            # 计算jerk（加加速度）
            if len(states_array) > 1:
                accel_diff = np.diff(states_array[:, :, 1], axis=0)
                jerk = accel_diff / self.simulation_step
                avg_jerk = np.mean(np.abs(jerk), axis=0)
            else:
                avg_jerk = np.zeros(3)

            # 稳定性指标
            spacing_std = np.std(spacings_array, axis=0)
            spacing_error = np.mean(np.abs(spacings_array - self.desired_spacing), axis=0)

            # 总奖励
            total_rewards = np.sum(rewards_array, axis=0)

            return {
                # 安全性
                'min_spacing': min_spacings.tolist(),
                'ttc_below_3_percent': ttc_below_3.tolist(),
                'collision_count': int(collision_count),
                # 效率性
                'avg_speed': avg_speeds.tolist(),
                'speed_std': speed_std.tolist(),
                'speed_following_error': speed_following_error.tolist(),
                'avg_leader_speed': float(avg_leader_speed),
                # 舒适性
                'accel_rms': accel_rms.tolist(),
                'avg_jerk': avg_jerk.tolist(),
                'avg_accel_abs': avg_accel_abs.tolist(),
                # 稳定性
                'spacing_std': spacing_std.tolist(),
                'spacing_error': spacing_error.tolist(),
                # 奖励
                'total_reward': total_rewards.tolist(),
                'avg_reward': np.mean(rewards_array, axis=0).tolist(),
                # 基本信息
                'episode_length': len(self.episode_data['time']),
                'episode_time': self.current_time,
                'total_steps': self.step_count
            }

        except Exception as e:
            print(f"⚠️ 计算性能指标失败：{e}")
            return {}

    def print_performance_summary(self):
        """打印性能摘要"""
        metrics = self.get_performance_metrics()
        if not metrics:
            print("❌ 无性能数据可显示")
            return

        print("\n" + "=" * 80)
        print("📊 CACC编队性能评估摘要")
        print("=" * 80)

        # 总体信息
        print(f"\n📋 总体信息：")
        print(f"  仿真时长：{metrics['episode_time']:.1f}秒")
        print(f"  总步数：{metrics['total_steps']}")
        print(f"  碰撞次数：{metrics['collision_count']}")
        print(f"  领头车平均速度：{metrics['avg_leader_speed']:.2f}m/s")

        # 安全性
        print(f"\n🛡️ 安全性指标：")
        for i, (min_space, ttc_pct) in enumerate(zip(metrics['min_spacing'], metrics['ttc_below_3_percent'])):
            follower_id = self.follower_ids[i]
            if min_space > self.safe_spacing:
                status = "✅ 安全"
            elif min_space > 3.0:
                status = "⚠️ 警告"
            else:
                status = "❌ 危险"
            print(f"  {follower_id}：最小间距={min_space:.2f}m ({status})，TTC<3s比例={ttc_pct:.1f}%")

        # 效率性
        print(f"\n⚡ 效率性指标：")
        for i, (avg_speed, speed_err) in enumerate(zip(metrics['avg_speed'], metrics['speed_following_error'])):
            follower_id = self.follower_ids[i]
            if speed_err < 1.0:
                status = "✅ 优秀"
            elif speed_err < 3.0:
                status = "⚠️ 良好"
            else:
                status = "❌ 较差"
            print(f"  {follower_id}：平均速度={avg_speed:.2f}m/s，跟驰误差={speed_err:.2f}m/s ({status})")

        # 舒适性
        print(f"\n😌 舒适性指标：")
        for i, (accel_rms, avg_jerk) in enumerate(zip(metrics['accel_rms'], metrics['avg_jerk'])):
            follower_id = self.follower_ids[i]
            if accel_rms < 0.5:
                status = "✅ 极佳"
            elif accel_rms < 1.0:
                status = "⚠️ 良好"
            else:
                status = "❌ 较差"
            print(f"  {follower_id}：加速度RMS={accel_rms:.3f}，加加速度均值={avg_jerk:.3f} ({status})")

        # 稳定性
        print(f"\n📈 稳定性指标：")
        for i, (spacing_std, spacing_err) in enumerate(zip(metrics['spacing_std'], metrics['spacing_error'])):
            follower_id = self.follower_ids[i]
            if spacing_std < 1.0:
                status = "✅ 优秀"
            elif spacing_std < 2.0:
                status = "⚠️ 良好"
            else:
                status = "❌ 较差"
            print(f"  {follower_id}：间距标准差={spacing_std:.2f}m，间距误差={spacing_err:.2f}m ({status})")

        # 奖励
        print(f"\n🏆 奖励汇总：")
        total_reward = sum(metrics['total_reward'])
        print(f"  总奖励：{total_reward:.2f}")
        for i, reward in enumerate(metrics['total_reward']):
            print(f"  {self.follower_ids[i]}：{reward:.2f}")

        print("=" * 80)

    def save_episode_data(self, filename=None):
        """保存回合数据到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"episode_data_{timestamp}.npz"

        try:
            data = {
                'time': np.array(self.episode_data['time']),
                'states': np.array(self.episode_data['states']),
                'actions': np.array(self.episode_data['actions']),
                'rewards': np.array(self.episode_data['rewards']),
                'leader_speeds': np.array(self.episode_data['leader_speeds']),
                'spacings': np.array(self.episode_data['spacings']),
                'collisions': np.array(self.episode_data['collisions']),
                'metrics': self.get_performance_metrics()
            }

            np.savez(filename, **data)
            print(f"💾 回合数据已保存：{filename}")
            return filename

        except Exception as e:
            print(f"❌ 保存数据失败：{e}")
            return None

    def close(self):
        """关闭环境"""
        print("\n📌 关闭仿真环境...")
        if self.is_connected:
            try:
                traci.close()
                self.is_connected = False
                print("✅ 环境已成功关闭")
            except:
                print("⚠️ 关闭连接时发生错误")


# 测试代码（验证环境功能）
if __name__ == "__main__":
    def test_env():
        """测试环境完整性"""
        print("=== CACC实时环境完整测试 ===")

        # 配置环境（启用GUI便于观察）
        config = {
            'sumo_config': 'straight.sumocfg',
            'gui': True,
            'step_length': 0.1,
            'collision_check': True,
            'max_steps_per_episode': 500
        }

        env = CACCRealTimeEnv(config)

        try:
            # 1. 重置环境
            print("\n1. 测试环境重置...")
            states = env.reset()
            print(f"   成功获取{len(states)}个跟随车辆的状态")

            # 2. 测试状态有效性
            print("\n2. 测试状态获取...")
            for i, (follower_id, state) in enumerate(zip(env.follower_ids, states)):
                print(f"   {follower_id}状态：{state[:3]}...（9维完整）")

            # 3. 测试仿真步进
            print("\n3. 测试仿真步进（100步）...")
            total_reward = 0
            for step in range(100):
                # 生成随机动作（小幅探索）
                actions = np.random.uniform(-0.3, 0.3, 3).tolist()
                next_states, rewards, done, info = env.step(actions)

                total_reward += np.mean(rewards)

                if step % 20 == 0:
                    print(
                        f"   步{step:3d}：时间={info['time']:.1f}s，奖励={np.mean(rewards):.3f}，间距={np.mean(info['spacings']):.2f}m")

                if done:
                    print(f"   仿真提前终止于步{step}")
                    break

            # 4. 测试性能指标
            print("\n4. 测试性能指标...")
            metrics = env.get_performance_metrics()
            env.print_performance_summary()

            # 5. 测试数据保存
            print("\n5. 测试数据保存...")
            env.save_episode_data()

            print("\n✅ 环境测试全部完成！")
            return True

        except Exception as e:
            print(f"\n❌ 测试失败：{e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            env.close()


    # 运行测试
    test_env()