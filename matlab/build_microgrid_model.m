%% build_microgrid_model.m
% Programmatically builds the Simulink model described in Project MD
% File.md (step 1), with the physically-broken DC-side transformer
% replaced by a DC-DC Buck Converter per OptionalUpgradesForFinalVersion.md
% section 2. Run this script in MATLAB (tested against R2023b+, Simulink +
% Simscape Electrical) to generate the .slx file; it is not auto-run or
% verified by this repo since MATLAB is not available in this environment.
%
% Topology:
%   Block B (6x 9V battery units, parallel)  --\
%                                                >-- Switching Apparatus + CU --- Grid (DC bus)
%   Block SC (6x 9V solar cell units, parallel)-/                                   |
%                                                        -------------------------------
%                                                        |                             |
%                                              DC-DC Buck Converter          (direct tap)
%                                                        |                             |
%                                              DC-AC Converter (Consumer)   DC-AC Converter (Industry)
%                                                        |                             |
%                                         Consumer Blocks 1-3 (1-5V)     Industry Blocks 1-3 (5-9V)

modelName = 'microgrid_infrastructure';
close_system(modelName, 0);
new_system(modelName);
open_system(modelName);

%% --- Block B: 6x 9V battery units in parallel ---
for i = 1:6
    name = sprintf('Battery_B%d', i);
    add_block('simscape/Electrical/Sources/Battery', [modelName '/' name], ...
        'Position', [50, 40*i, 130, 40*i + 30]);
end

%% --- Block SC: 6x 9V solar cell units in parallel ---
for i = 1:6
    name = sprintf('SolarCell_SC%d', i);
    add_block('simscape/Electrical/Sources/Solar Cell', [modelName '/' name], ...
        'Position', [50, 300 + 40*i, 130, 300 + 40*i + 30]);
end

%% --- Switching Apparatus + Controller Unit (CU) ---
add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/Switching_Apparatus'], ...
    'Position', [220, 200, 340, 320]);
add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/Controller_Unit_CU'], ...
    'Position', [220, 340, 340, 420]);
% CU is populated separately by controller_stateflow.m (hysteresis) or by
% importing the ONNX-exported PyTorch linear-regression model via the Deep
% Learning Toolbox block, per OptionalUpgradesForFinalVersion.md section 3.

%% --- Grid DC bus ---
add_block('simulink/Signal Routing/Bus Creator', [modelName '/Grid_DC_Bus'], ...
    'Position', [400, 240, 430, 280]);

%% --- DC-DC Buck Converter (replaces the invalid DC-side transformer) ---
add_block('simscape/Electrical/Power Converters/DC-DC Converters/Buck Converter', ...
    [modelName '/Buck_Converter'], 'Position', [480, 320, 580, 360]);
set_param([modelName '/Buck_Converter'], 'OutputVoltage', '5'); % regulated 3-7V band, nominal 5V

%% --- Two DC-AC Converter setups ---
add_block('simscape/Electrical/Power Converters/Inverters/Averaged Inverter', ...
    [modelName '/DCAC_Consumer'], 'Position', [640, 320, 740, 360]);
add_block('simscape/Electrical/Power Converters/Inverters/Averaged Inverter', ...
    [modelName '/DCAC_Industry'], 'Position', [640, 160, 740, 200]);

%% --- Load blocks ---
for i = 1:3
    name = sprintf('Consumer_Block_%d', i);
    add_block('simscape/Electrical/Passive/Variable Resistor', [modelName '/' name], ...
        'Position', [800, 320 + 40*i, 880, 320 + 40*i + 30]);
end
for i = 1:3
    name = sprintf('Industry_Block_%d', i);
    add_block('simscape/Electrical/Passive/Variable Resistor', [modelName '/' name], ...
        'Position', [800, 160 + 40*i, 880, 160 + 40*i + 30]);
end

%% --- Wiring ---
% NOTE: add_line calls below assume default port names/positions; verify
% and adjust port indices interactively once blocks are placed, since exact
% port counts vary by Simscape Electrical version.
add_line(modelName, 'Grid_DC_Bus/1', 'Buck_Converter/1', 'autorouting', 'on');
add_line(modelName, 'Grid_DC_Bus/1', 'DCAC_Industry/1', 'autorouting', 'on');
add_line(modelName, 'Buck_Converter/1', 'DCAC_Consumer/1', 'autorouting', 'on');

save_system(modelName, fullfile(pwd, [modelName '.slx']));
fprintf('Saved %s.slx\n', modelName);
