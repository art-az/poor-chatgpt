def split_file(input_file, output_file1, output_file2, output_file3, output_file4):
    # Open the original file and read all lines
    with open(input_file, 'r') as file:
        text = file.read()  # Split the file by words (whitespace)

      # Split text into sentences based on period and whitespace
    sentences = text.split('.')
    
    # Clean up extra whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Find the halfway point
    half_point = len(sentences) // 4
    

    # Write the first half to output_file1
    with open(output_file1, 'w') as file1:
        for sentence in sentences[:half_point]:
            file1.write(sentence + '.\n')  # Re-add the period and newline at the end of each sentence

    # Write the second half to output_file2
    with open(output_file2, 'w') as file2:
        for sentence in sentences[:half_point]:
            file2.write(sentence + '.\n')  # Re-add the period and newline at the end of each sentence

    with open(output_file3, 'w') as file3:
        for sentence in sentences[:half_point]:
            file3.write(sentence + '.\n')  # Re-add the period and newline at the end of each sentence

    with open(output_file4, 'w') as file4:
        for sentence in sentences[:half_point]:
            file4.write(sentence + '.\n')  # Re-add the period and newline at the end of each sentence

# Example usage:
split_file('en.txt', 'en_first_quarter.txt', 'en_second_quarter.txt', 'en_third_quarter.txt', 'en_forth_quarter.txt')