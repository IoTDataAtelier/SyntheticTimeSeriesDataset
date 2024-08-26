import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np 
import os

class WaveGenerator:
    def __init__(self, start: int = 0, stop: int = 10, data_number: int = 160000) -> None:
        self.data_ponits = np.linspace(start=start, stop=stop, num=data_number)
        pass

    def add_noise(self, multiple_time_series):
        time_series_with_noise = []
        noise = np.random.normal(0,0.1, len(multiple_time_series[0]))
        for time_series in multiple_time_series:
            time_series_with_noise.append(time_series + noise)

        return time_series_with_noise    
    
    def generate_time_series(self, angular_frequency: float):
        return np.sin(angular_frequency*self.data_ponits) + 0.1*np.random.rand(len(self.data_ponits))
    
    def mixing_time_series(self, first_time_series, second_time_series, coeficient_pi = 2*np.pi):
        sine_wave = np.sin(coeficient_pi*self.data_ponits)
        return first_time_series*sine_wave + (1-sine_wave)*second_time_series

    def plot_wave(self, serie, w = 16000, label_hz = 'Hz', color='b'):
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(serie[:2*w], c=color, label=label_hz)
        ax[0].legend()
        ax[1].plot(np.log(np.abs(np.fft.fft(serie[:w])))[:int(w/2)], c=color, label=label_hz)
        ax[1].legend()
        plt.legend()

    def generate_wav_file(self, file_name, serie, folder_name = 'syntheticWaves', sampling_rate = 16000, amount=10):
        path = './'+folder_name

        if not os.path.exists(path):
            os.mkdir(path)

        initial_index = 0
        final_index = sum_coeficient = int(len(serie)/amount)

        for i in range(amount):
            sf.write(path+'/'+file_name + str(i)+ '.wav', serie[initial_index:final_index], sampling_rate)
            initial_index = final_index
            final_index += sum_coeficient
