"""
Smoke test for DDPG implementation with Stable-Baselines-3.

Quick validation that:
1. Environment creation works
2. DDPG agent initializes correctly
3. Training loop executes without errors
4. Model saving/loading works
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Add project to path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "RLI_17_A0"))
os.chdir(str(project_dir / "RLI_17_A0"))

import gym_race

# Import DDPG training module
from dqn_vanilla.Pyrace_RL_DDPG_SB3 import ContinuousActionWrapper


def test_continuous_environment():
    """Test 1: Continuous action wrapper works"""
    print("\n" + "=" * 60)
    print("TEST 1: Continuous Environment Creation")
    print("=" * 60)
    
    try:
        env = gym.make("Pyrace-v3")
        env = ContinuousActionWrapper(env)
        
        print(f"✓ Environment created: {env.spec.id}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")
        print(f"  Initial obs shape: {obs.shape}")
        print(f"  Initial obs: {obs}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Action taken: {action}")
        print(f"  Reward: {reward:.4f}")
        
        env.close()
        print("\n✅ TEST 1 PASSED: Environment works")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddpg_initialization():
    """Test 2: DDPG agent initialization"""
    print("\n" + "=" * 60)
    print("TEST 2: DDPG Agent Initialization")
    print("=" * 60)
    
    try:
        env = gym.make("Pyrace-v3")
        env = ContinuousActionWrapper(env)
        
        # Create action noise
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        # Create DDPG model
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=100,  # Small buffer for testing
            batch_size=32,
            action_noise=action_noise,
            verbose=0
        )
        
        print(f"✓ DDPG model created")
        print(f"  Actor network: {model.actor}")
        print(f"  Critic network: {model.critic}")
        print(f"  Learning rate: {model.learning_rate}")
        print(f"  Batch size: {model.batch_size}")
        
        env.close()
        print("\n✅ TEST 2 PASSED: DDPG initialization works")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddpg_training_short():
    """Test 3: Brief DDPG training (10 steps)"""
    print("\n" + "=" * 60)
    print("TEST 3: DDPG Short Training (10 timesteps)")
    print("=" * 60)
    
    try:
        env = gym.make("Pyrace-v3")
        env = ContinuousActionWrapper(env)
        
        # Create action noise
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        # Create and train DDPG model
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=100,
            batch_size=32,
            action_noise=action_noise,
            verbose=0
        )
        
        print("Starting 10-step training...")
        model.learn(total_timesteps=10, progress_bar=False)
        
        print(f"✓ Training completed")
        print(f"  Number of steps: {model.num_timesteps}")
        print(f"  Replay buffer pos: {model.replay_buffer.pos}")
        
        env.close()
        print("\n✅ TEST 3 PASSED: Training loop works")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """Test 4: Model saving and loading"""
    print("\n" + "=" * 60)
    print("TEST 4: Model Save/Load")
    print("=" * 60)
    
    try:
        model_dir = Path("dqn_vanilla/models_DDPG_smoke_test")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "test_ddpg"
        
        env = gym.make("Pyrace-v3")
        env = ContinuousActionWrapper(env)
        
        # Create and save model
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            action_noise=action_noise,
            verbose=0
        )
        
        model.save(str(model_path))
        print(f"✓ Model saved to {model_path}.zip")
        
        # Verify file exists
        assert (model_dir / f"{model_path.name}.zip").exists()
        
        # Load model
        loaded_model = DDPG.load(str(model_path))
        print(f"✓ Model loaded successfully")
        
        # Verify loaded model works
        obs, _ = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        print(f"✓ Loaded model can predict actions")
        print(f"  Predicted action: {action}")
        
        env.close()
        print("\n✅ TEST 4 PASSED: Save/load works")
        return True
    
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    print("\n" + "=" * 60)
    print("DDPG SB3 Smoke Test Suite")
    print("=" * 60)
    
    results = {
        "Environment Creation": test_continuous_environment(),
        "DDPG Initialization": test_ddpg_initialization(),
        "Short Training": test_ddpg_training_short(),
        "Save/Load": test_model_save_load(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} | {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! DDPG implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Fix before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
