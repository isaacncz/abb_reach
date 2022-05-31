import os

import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


from datetime import datetime

import telegram
token = ""
chat_id = 0
with open('src/config.txt') as f:
    lines = f.readline().split(',')
token = lines[0]
chat_id = int(lines[1])
bot = telegram.Bot(token=token)




class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1,device="1"):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.logNumber = 0
        self.device = device

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        path = "/home/isaac/irb120_ws/src/openai_abb/src/tensorboard"
        dir_list = os.listdir(path)
        allNumber = []
        for i in dir_list:
            x = i.split("_")
            number = int(x[1])
            allNumber.append(number)
        self.logNumber = max(allNumber) 

        model_param = ""
        model_param = "Training starts at device (" + self.device + ") " + datetime.now().strftime("%b-%d_%H:%M:%S")
        model_param += "\nTQC_" + str(self.logNumber)
        model_param += "\nlearning_rate:" + str(self.model.learning_rate)
        model_param += "\nbuffer_size:" + str(self.model.buffer_size)
        model_param += "\nbatch_size:" + str(self.model.batch_size)
        model_param += "\ntau:" + str(self.model.tau)
        model_param += "\ngamma:" + str(self.model.gamma)
        # model_param += "\ntop_quantiles_to_drop_per_net:" + str(self.model.top_quantiles_to_drop_per_net)
        model_param += "\npolicy_kwargs:" + str(self.model.policy_kwargs)
        # model_param += "\naction_noise:" + str(self.model.action_noise)
        model_param += "\nreplay_buffer_kwargs:" + str(self.model.replay_buffer_kwargs)
        print(model_param)
        bot.send_message(text=model_param, chat_id=chat_id)
        # message = str(self.model.observation_space)
        # bot.send_message(text=message, chat_id=chat_id)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        message = "Training for TQC_" + str(self.logNumber)+   " ends at " + datetime.now().strftime("%b-%d_%H:%M:%S")
        bot.send_message(text=message, chat_id=452439053)


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                # message = "Num timesteps: {self.num_timesteps} Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                message = "Device (" + self.device + ")\n " +"Timesteps: " + str(self.num_timesteps) + '\nBest mean: ' + str(self.best_mean_reward) + '\nLast mean:' + str(mean_reward)
                bot.send_message(text=message, chat_id=chat_id)
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                    # message = "Saving new best model"
                    # bot.send_message(text=message, chat_id=chat_id)
                  self.model.save(self.save_path)
                  self.model.save_replay_buffer(os.path.join(self.log_dir,'best_replay_buffer'))

        return True


