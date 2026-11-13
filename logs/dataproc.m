clear;
[logFile, logDir] = uigetfile('teleop_*.csv', '选择遥操作日志');
if isequal(logFile, 0)
    clear logFile logDir
    return
end

% 按 CSV 表头读取，保留每一个采样点。
logPath = fullfile(logDir, logFile);
logOpts = detectImportOptions(logPath, 'Delimiter', ',', 'Encoding', 'UTF-8', ...
    'VariableNamesLine', 1, 'VariableNamingRule', 'preserve');
logOpts.DataLines = [2 Inf];
logNames = logOpts.VariableNames;

% 全部字段按数值导入，缺失采样保留为 NaN。
logOpts = setvartype(logOpts, logNames, 'double');
logOpts = setvaropts(logOpts, logNames, 'TreatAsMissing', {'NULL', 'nan', 'NaN'});
logOpts.MissingRule = 'fill';
logTable = readtable(logPath, logOpts);

for logIndex = 1:numel(logNames)
    logName = logNames{logIndex};
    logParts = regexp(logName, '^(master|slave)_j([1-6])_(.+)$', 'tokens', 'once');

    if isempty(logParts)
        % 非六轴字段直接使用原名。
        assignin('base', logName, logTable.(logName));

    elseif strcmp(logParts{2}, '1')
        % 遇到 J1 时，一次取出同一字段的 J1～J6，按轴号排成六列。
        logGroup = [logParts{1} '_' logParts{3}];
        logColumns = cell(1, 6);
        for logAxis = 1:6
            logColumns{logAxis} = sprintf('%s_j%d_%s', ...
                logParts{1}, logAxis, logParts{3});
        end
        assignin('base', logGroup, logTable{:, logColumns});
    end
end

clear logFile logDir logPath logOpts logNames logTable
clear logIndex logName logParts logGroup logColumns logAxis

%%

master_slave_qd=master_q_rad-slave_q_rad;
master_slave_dqd=master_dq_rad_s-slave_dq_rad_s;
k=[40,40,40,20,20,20];
b=[10, 10, 10, 5, 5, 5]*0.5;
fd_torque=master_slave_qd.*k-master_slave_dqd.*b;
fd_torque_nodamping=master_slave_qd.*k;
figure(1);clf(1);
plot(fd_torque(:,3));
hold on;
plot(fd_torque_nodamping(:,3));
legend([{'damping'},{'no damping'}]);

figure(2);clf(2);
plot(master_slave_dqd(:,3));
hold on;
% 主从反馈有效时持续记录滤波值。图中取相反数，使两条曲线均为主端减从端。
plot(-master_filtered_velocity_error_rad_s(:,3));
ylabel('J3 主端减从端角速度 (rad/s)');
legend('原始速度差', '滤波后速度差');
