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
figure(1);clf(1);
plot(master_slave_qd(:,3));