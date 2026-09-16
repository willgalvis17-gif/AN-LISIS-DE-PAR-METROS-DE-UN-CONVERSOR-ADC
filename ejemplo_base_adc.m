%% ========================================================================
%% Ejemplo Base: Lectura y Graficacion de Datos del ADC (MATLAB)
%% ========================================================================
clear; clc; close all;

% Asegura trabajar en la carpeta donde esta guardado este script
try, cd(fileparts(mfilename('fullpath'))); catch, end

%% 1. Grafica en el Dominio del Tiempo
data_t = readtable('adc_samples_250p0Hz_hann.csv');

figure('Color', 'w');
subplot(2,1,1);
plot(data_t.time_s * 1000, data_t.voltage_V, 'b.-');
grid on;
xlabel('Tiempo [ms]'); ylabel('Voltaje [V]');
title('Senal Adquirida en el Dominio del Tiempo');
xlim([0, 20]); % Primeros 20 ms (~5 ciclos a 250 Hz)

%% 2. Grafica del Espectro en Frecuencia (dBFS)
data_f = readtable('adc_fft_250p0Hz_hann.csv');

subplot(2,1,2);
plot(data_f.frequency_Hz, data_f.magnitude_dBFS, 'Color', [0.4 0.4 0.4]);
grid on;
xlabel('Frecuencia [Hz]'); ylabel('Magnitud [dBFS]');
title('Espectro en Frecuencia (FFT)');
xlim([0, 1000]); % Banda de Nyquist (Fs/2)
ylim([-120, 5]);


