import spacy

def list_similarity(A, input_list, threshold=0.7):
    nlp = spacy.load('en_core_web_md')
    # print(A)
    # print(input_list)
    # exit()
    # Process sentences in list A
    doc_list_A = [nlp(sent) for sent in A]
    
    # Process input sentences
    doc_list_input = [nlp(sent) for sent in input_list]
    
    # Calculate similarity scores between input sentences and sentences in list A
    similarity_scores = []
    for input_doc in doc_list_input:
        doc_similarities = [input_doc.similarity(sent) for sent in doc_list_A]
        max_similarity = max(doc_similarities)
        similarity_scores.append(max_similarity)
    
    # Calculate average similarity score
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    
    return avg_similarity

# A = ["Quantum mechanics is a branch of physics that deals with the behavior of matter and energy at the atomic and subatomic level.",     "It is known for the wave-particle duality, which states that particles can exhibit properties of both waves and particles.",     "The uncertainty principle states that certain properties of a particle cannot be known simultaneously with perfect accuracy.",     "In quantum mechanics, a system is described by a wave function, also known as a quantum state, which is governed by the Schrödinger equation."]
     
# input_list = ["Quantum mechanics, branch of physics that deals with the behavior of matter and energy at the atomic and subatomic level.",              "It is known for the wave-particle duality, which states that particles can exhibit properties of both waves and particles.",              "The uncertainty principle states that certain properties of a particle cannot be known simultaneously with perfect accuracy.",              "In quantum mechanics, a system is described by a wave function, also known as a quantum state, which is governed by the Schrödinger equation."]

# similarity_score = list_similarity(A, input_list, threshold=0.6)

# print("The similarity score between the two lists is: ", similarity_score)
