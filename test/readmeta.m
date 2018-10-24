[filename, activity, session] = textread('meta.txt','%s %s %s'  );
presentFiles = dir( '/HOMES/kreuzer/Downloads/DCASE2018-task5-dev/audio' );
presentFiles= { presentFiles.name }.';
presentFiles = strcat('audio/',presentFiles);
activity_tags = {'absence';'other';'vaccum_cleaning';'working';'watching_tv';'visit';'eating';'dishwashing';'cooking';'phone_call'};

for i=1:length(activity_tags)
    indices = find(strcmp(activity, activity_tags{i}));
    activity_filenames = filename(indices);
    files{i} = intersect( activity_filenames,presentFiles);
    node1_idx = find(contains(files{i},'Node1'));
    node2_idx = find(contains(files{i},'Node2'));
    node3_idx = find(contains(files{i},'Node3'));
    node4_idx = find(contains(files{i},'Node4'));
    file_names_sorted{i} = {files{i}(node1_idx),files{i}(node2_idx),files{i}(node3_idx),files{i}(node4_idx)};
end

%idx = find(ismember(C, 'bla'))
% node1_idx_absence = find(contains(files{2},'Node1'));
% absence_node1 = files{1}(node2_idx_absence);
% node2_idx_absence = find(contains(files{2},'Node2'));
% absence_node2 = files{2}(node2_idx_absence);

%idx = find(ismember(C, 'bla'))