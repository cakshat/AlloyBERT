import torch

#--------------------------------------------------------------------------------
# We need to find where each word starts and finishes after tokenization.
# Some words will break down during tokenization (like "going" to "go"+"ing"), and
# we need to loop through the token_ids and say (for example) indexes 10 and 11  
# corresponds with the word "going". We do the same thing to find the locations 
# of [dot] tokens and stop words as well.
#
# Then we can use these positions to either calculate the score of a multi-part
# word, or ignore the [dot] tokens.
#
# For more information about it, read the blog post mentioned in the README.md
#--------------------------------------------------------------------------------
def find_positions(ignore_specials, ignore_stopwords, the_tokens, stop_words):
    dot_positions = {}
    stopwords_positions = {}
    tmp = []

    if ignore_specials:
        word_counter = 0
        start_pointer = 0
        positions = {}

        num_of_tokens = len( the_tokens )
        num_of_tokens_range = range( num_of_tokens + 1 )

    else:
        word_counter = 1
        start_pointer = 1
        positions = {0: [0, 1]}

        num_of_tokens = len( the_tokens ) - 1
        num_of_tokens_range = range( 1, num_of_tokens + 1 )


    for i in num_of_tokens_range:

        if i == num_of_tokens:
            positions[word_counter] = [start_pointer, i]
            break

        if the_tokens[i][0] in ['Ġ', '.']:

            if ignore_stopwords:
                joined_tmp = "".join(tmp)
                current_word = joined_tmp[1:] if joined_tmp[0] == "Ġ" else joined_tmp
                if current_word in stop_words:
                    stopwords_positions[word_counter] = i-1

            if the_tokens[i] == ".":
                dot_positions[word_counter+1] = i

            positions[word_counter] = [start_pointer, i]
            word_counter += 1
            start_pointer = i
            tmp = []

        tmp.append(the_tokens[i])

    if not ignore_specials:
        positions[len( positions )] = [i, i+1]
    
    return positions, dot_positions, stopwords_positions

#--------------------------------------------------------------------------------
# Splitting the text into words as a refrence. Then we can map the words to the 
# find_positions() function's output.
#--------------------------------------------------------------------------------
def make_the_words(inp, positions, ignore_specials):
    num_of_words = len( positions )

    if ignore_specials:
        the_words = inp.replace(".", " .").split(" ")[0:num_of_words]

    else:
        the_words = inp.replace(".", " .").split(" ")[0:(num_of_words-2)]
        the_words = ['[BOS]'] + the_words + ['[EOS]']
    
    return the_words

#--------------------------------------------------------------------------------
# A min-max normalizer! We use it to normalize the scores after ignoring some tokens.
#--------------------------------------------------------------------------------
def scale(x, min_, max_):
    return (x - min_) / (max_ - min_)

#--------------------------------------------------------------------------------
# A helper function to use the scores and display each word color-coded.
#--------------------------------------------------------------------------------
def make_html(the_words, positions, final_score, num_words=15):
    the_html = ""

    for i, word in enumerate( the_words ):
        if i in positions:
            start = positions[i][0]
            end   = positions[i][1]

            if end - start > 1:
                score = torch.max( final_score[start:end] )
            else:
                score = final_score[start]

            the_html += """<span style="background-color:rgba(255, 0, 0, {});
                        padding:3px 6px 3px 6px; margin: 0px 2px 0px 2px" title="{}">{}</span>""" \
                        .format(score, score, word)

        if ((i+1) % num_words) == 0:
            the_html += "<br />"

    return the_html

#--------------------------------------------------------------------------------
# Returns a sample article if package is initialized witt "with_sample=True" argument.
#--------------------------------------------------------------------------------
def get_sample_article():
    return """"Test temperature: 1922.038889 K.Solidus temperature: 3650.82516 K.W composition: 0.9703249374307252. Tungsten (Atomic # 74, Weight 183.84, Young's Modulus 411 GPa, Microstructure BCC, Melting Point 3422 °C)Nb composition: 1.9794514598817563e-06. Niobium (Atomic # 41, Weight 92.91, Young's Modulus 105 GPa, Microstructure BCC, Melting Point 2468 °C)Hf composition: 1.030330379983916e-06. Hafnium (Atomic # 72, Weight 178.49, Young's Modulus 78 GPa, Microstructure HCP, Melting Point 2233 °C)Mo composition: 1.916661485391654e-06. Molybdenum (Atomic # 42, Weight 95.95, Young's Modulus 329 GPa, Microstructure BCC, Melting Point 2623 °C)Re composition: 0.0296289080738096. Rhenium (Atomic # 75, Weight 186.21, Young's Modulus 463 GPa, Microstructure HCP, Melting Point 3186 °C)Zr composition: 2.015957089398943e-06. Zirconium (Atomic # 40, Weight 91.22, Young's Modulus 98 GPa, Microstructure HCP, Melting Point 1855 °C)Ta composition: 1.016335032625578e-06. Tantalum (Atomic # 73, Weight 180.95, Young's Modulus 186 GPa, Microstructure BCC, Melting Point 2996 °C)Ti composition: 3.841971912242864e-06. Titanium (Atomic # 22, Weight 47.87, Young's Modulus 116 GPa, Microstructure HCP, Melting Point 1668 °C)C composition: 1.531127046235361e-05. Carbon (Atomic # 6, Weight 12.01, Young's Modulus Varies, Microstructure Varies, Melting Point Sublimation around 3915 °C for diamond)Y composition: 2.0685215823630188e-06. Yttrium (Atomic # 39, Weight 88.91, Young's Modulus 63 GPa, Microstructure HCP, Melting Point 1799 °C)Al composition: 6.8159073329111865e-06. Aluminum (Atomic # 13, Weight 26.98, Young's Modulus 69 GPa, Microstructure FCC, Melting Point 660.3 °C)Si composition: 6.547993431604535e-06. Silicon (Atomic # 14, Weight 28.09, Young's Modulus 130 GPa, Microstructure Diamond cubic, Melting Point 1414 °C)V composition: 3.610095296042111e-06. Vanadium (Atomic # 23, Weight 50.94, Young's Modulus 128 GPa, Microstructure BCC, Melting Point 1910 °C)"""