def main():
    file1 = open('git_revert.txt', 'r')
    lines = file1.readlines()
    with open('git_commands.txt', 'w') as file2:
        for line in lines:
            new_line = line[12:]
            command = "git checkout -- " + new_line
            file2.write(command)

if __name__ == '__main__':
    main()