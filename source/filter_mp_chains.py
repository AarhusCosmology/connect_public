import numpy as np

def keep_last_N_lines(list_of_files, N):
    output_lines = {}
    #total_lines = 0
    lines_of_use_in_files = {}
    for filename in list_of_files:
        with open(filename, 'r') as f:
            lines = 0
            for i, line in enumerate(f, 1):
                if line[0] != '#':
                    #total_lines += 1
                    lines += 1
                else:
                    print(f'updated at line {i}')
                    lines = 0
        lines_of_use_in_files[filename] = lines
    print(lines_of_use_in_files)

    N_per_file = int(np.ceil(N/len(list_of_files)))

    #ratio = N/total_lines
    if sum(lines_of_use_in_files.values()) <= N:
        lines_to_be_used = lines_of_use_in_files
    else:
        if any([lines_of_use_in_files[filename] < N_per_file for filename in list_of_files]):
            files_short = [filename for filename in list_of_files if lines_of_use_in_files[filename] < N_per_file]
            lines_missing = sum([N_per_file - lines_of_use_in_files[filename] for filename in files_short])
            lines_missing -= N_per_file*len(list_of_files) - N

            lines_to_be_used = {}
            for filename in list_of_files:
                if filename in files_short:
                    lines_to_be_used[filename] = lines_of_use_in_files[filename]
                else:
                    lines_to_be_used[filename] = N_per_file
            
            stop = False
            while not stop:
                for filename in list_of_files:
                    if not filename in files_short:
                        if lines_of_use_in_files[filename] > lines_to_be_used[filename]:
                            lines_to_be_used[filename] += 1
                            lines_missing -= 1
                            if not lines_missing > 0:
                                stop = True
                                break

        else:
            lines_to_be_used = {}
            for filename in list_of_files:
                lines_to_be_used[filename] = N_per_file

    for filename, N_lines in lines_to_be_used.items():
        with open(filename, 'r') as f:
            f_list = list(f)
        #n = 0
        output_lines[filename] = []
        for i, line in enumerate(reversed(f_list), 1):
            if i <= N_lines:
                output_lines[filename].append(line)

            #if n >= N/len(list_of_files):#ratio*N_lines:
            #    break
            #elif line[0] != '#':
            #    output_lines[filename].append(line)
            #    n += 1
            #else:
            #    break
    """
    N_out = sum(lines_to_be_used.values())
    #for filename in list_of_files:
    #    N_out += len(output_lines[filename])
    if N_out > N:
        diff = int(N_out-N)
        output_lines[list_of_files[0]] = output_lines[list_of_files[0]][:-diff]
    """
    for filename in list_of_files:
        with open(filename, 'w') as f:
            for line in output_lines[filename]:
                f.write(line)
    
