/**
 * @file glue.cpp
 * @brief Context aware document chunker
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include "bert.h"

#define EMBEDDING_DIM 768
#define DEFAULT_THRESHOLD 0.9
#define DEFAULT_MAX_CHUNK_SIZE 250
#define DEFAULT_MIN_CHUNK_SIZE 50
#define DEFAULT_OVERLAP 1
#define OUTPUT_PATH "glue-output.json"

using ChunkValue = std::variant<std::string, int, std::vector<float>>;


class Chunk
{
private:
    std::string text;
    int size;
    int seq;
    std::vector<float> embedding;

public:
    Chunk(const std::string &text, const int seq, const std::vector<float> &embedding)
    {
        this->text = text;
        this->size = text.size();
        this->seq = seq;
        this->embedding = embedding;
    }

    std::string __str__()
    {
        return "Chunk(text=" + this->text + ", seq=" + std::to_string(this->seq) + ")";
    }

    std::unordered_map<std::string, ChunkValue> to_dict() const {
        return std::unordered_map<std::string, ChunkValue> {
            {"text", text},
            {"size", size},
            {"seq", seq},
            {"embedding", embedding}
        };
    }

    // setter methods

    void set_vector(const std::vector<float> &embedding)
    {
        this->embedding = embedding;
    }

    // getter methods

    std::string get_text() { return this->text; }
    int get_size() { return this->size; }
    int get_seq() { return this->seq; }
    std::vector<float> get_vector() { return this->embedding; }
};


std::vector<float> embedding_provider(const std::string& text, bert_ctx* ctx){
    float* embeddings = new float[bert_n_embd(ctx)];
    bert_encode(ctx, 1, text.c_str(), embeddings);

    std::vector<float> result(embeddings, embeddings + bert_n_embd(ctx));
    delete[] embeddings;

    return result;
}


float cosine_similarity(std::vector<float> v1, std::vector<float> v2){
    float dot_product = 0.0;
    float norm_v1 = 0.0;
    float norm_v2 = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; i++)
    {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }
    return dot_product / (sqrt(norm_v1) * sqrt(norm_v2));

}


std::vector<std::string> init_text_chunker(const std::string &text)
{
    std::vector<std::string> sentences;
    std::string currentSentence;
    for (char ch : text)
    {
        currentSentence += ch;
        // Check if the current character is an end-of-sentence marker.
        if (ch == '.' || ch == '!' || ch == '?')
        {
            currentSentence.erase(currentSentence.find_last_not_of(" \n\r\t") + 1);
            sentences.push_back(currentSentence);
            currentSentence.clear();
        }
    }
    // Add any remaining text as the last sentence (if not empty).
    if (!currentSentence.empty())
    {
        // Trim spaces at the end (optional).
        currentSentence.erase(currentSentence.find_last_not_of(" \n\r\t") + 1);
        sentences.push_back(currentSentence);
    }
    return sentences;
}


std::vector<Chunk> embed_init_chunks(const std::vector<std::string> &sentences)
{
    bert_ctx* ctx = bert_load_from_file("bert.cpp/models/all-MiniLM-L6-v2/ggml-model-q4_0.bin");

    std::vector<Chunk> chunks;
    int n = sentences.size();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {   
        std::cout << "Processing sentence: " << i << std::endl;
        std::vector<float> embedding = embedding_provider(sentences[i], ctx);
        Chunk chunk = Chunk(sentences[i], i, embedding);
        chunks.push_back(chunk);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Initial Embedding Duration: " << elapsed.count() << " s" << std::endl;
    bert_free(ctx);
    return chunks;
}


std::vector<float> average_vectors(std::vector<float> v1, std::vector<float> v2){
    std::vector<float> avg_vector(v1.size(), 0.0);
    int n = v1.size();
    for (int i = 0; i < n; i++)
    {
        avg_vector[i] = (v1[i] + v2[i]) / 2;
    }
    return avg_vector;
}


std::vector<Chunk> glue(
    std::string text,
    float threshold,
    int max_chunk_size,
    int min_chunk_size,
    int overlap
){
    std::vector<Chunk> result = std::vector<Chunk>();
    std::vector<std::string> sentences = init_text_chunker(text);
    std::vector<Chunk> chunks = embed_init_chunks(sentences);
    int n = chunks.size();

    std::vector<std::vector<float>> similarity_matrix = std::vector<std::vector<float>>(n, std::vector<float>(n, -1.0));

    std::vector<Chunk> glued_chunks = std::vector<Chunk>();

    int seq = 0;
    for (int i = 0; i < n; i++)
    {

        std::vector<Chunk> curr = std::vector<Chunk>();
        curr.push_back(chunks[i]);
        
        int running_size = chunks[i].get_size();
        std::string running_text = chunks[i].get_text();
        std::vector<float> running_vector = chunks[i].get_vector();

        int j = i + 1;
        while (j < n)
        {
            if (similarity_matrix[i][j] == -1.0)
            {
                similarity_matrix[i][j] = cosine_similarity(chunks[i].get_vector(), chunks[j].get_vector());
            }

            if(
                (similarity_matrix[i][j] > threshold &&
                running_size + chunks[j].get_size() <= max_chunk_size) ||
                (running_size + chunks[j].get_size() <= min_chunk_size)
            ){
                curr.push_back(chunks[j]);
                running_size += chunks[j].get_size();
                running_text += chunks[j].get_text();
                running_vector = average_vectors(running_vector, chunks[j].get_vector());
            }else{
                break;
            }

            j++;
        }

        i = std::max(j - overlap, i + 1);
        Chunk chunk = Chunk(running_text, seq++, running_vector);
        result.push_back(chunk);        
    }
    return result;
}


std::string chunk_map_tostring(const ChunkValue& chunk){
    std::ostringstream oss;
    if (std::holds_alternative<std::string>(chunk)){
        oss << "\"" << std::get<std::string>(chunk) << "\"";
    } else if (std::holds_alternative<int>(chunk)){
        oss << std::get<int>(chunk);
    } else if (std::holds_alternative<std::vector<float>>(chunk)){
        oss << "[";
        const auto& vec = std::get<std::vector<float>>(chunk);
        int n = vec.size();
        for (int i = 0; i < n; i++){
            oss << vec[i];
            if (i < n - 1){
                oss << ", ";
            }
        }
        oss << "]";
    }
    return oss.str();
}


int output_json(const std::vector<Chunk>& chunks, const std::string& path){
    std::ofstream outFile(path, std::ios::app);
    if (!outFile.is_open()){
        std::cerr << "Failed to open output file: " << path << std::endl;
        return -1;
    }

    outFile << "[" << std::endl;
    for (auto it = chunks.begin(); it != chunks.end(); it++){
        const auto& chunk_dict = it->to_dict();
        outFile << " {" << std::endl;
        for (auto mapIt = chunk_dict.begin(); mapIt != chunk_dict.end(); mapIt++){
            outFile << "  \"" << mapIt->first << "\": " << chunk_map_tostring(mapIt->second);
            if (std::next(mapIt) != chunk_dict.end()){
                outFile << ",";
            }
            outFile << std::endl;
        }
        outFile << " }";
        if (std::next(it) != chunks.end()){
            outFile << ",";
        } 
        outFile << std::endl;
    }
    outFile << "]" << std::endl;
    outFile.close();
    return 0;
}



std::string help = 
"Usage: glue <input_file_path> [options]\n"
"Arguments:\n"
"  input_file_path             Path to the input file to be processed.\n"
"Options:\n"
"  -t, --threshold <value>     Set the similarity threshold for chunking (float).\n"
"                              This determines how similar text chunks need to be in order to be processed together.\n"
"  -x, --max_chunk_size <size> Set the maximum size of a chunk (integer).\n"
"                              This limits the maximum number of elements (e.g., words, characters) a single chunk can contain.\n"
"  -n, --min_chunk_size <size> Set the minimum size of a chunk (integer).\n"
"                              This specifies the minimum number of elements a chunk must contain to be considered valid.\n"
"  -o, --overlap <size>        Set the overlap size between chunks (integer).\n"
"                              This determines how many elements at the end of one chunk can be repeated at the beginning of the next chunk.\n"
"  -p, --output <output_path>  Specify the path to the output file.\n"
"                              If provided, the processed data will be written to this file. Otherwise, the output will be printed to stdout.\n";

int main(int argc, char* argv[])
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << help << std::endl;
        return EXIT_SUCCESS;
    }

    if (argc < 2) {
        std::cerr << help << std::endl;
        return EXIT_FAILURE; 
    }

    std::string input_fpath = argv[1];

    float threshold = DEFAULT_THRESHOLD;
    int max_chunk_size = DEFAULT_MAX_CHUNK_SIZE;
    int min_chunk_size = DEFAULT_MIN_CHUNK_SIZE;
    int overlap = DEFAULT_OVERLAP;
    std::string output_fpath = OUTPUT_PATH;

    for (int i = 2; i < argc; i++){
        std::string arg = argv[i];
        if ((arg == "--threshold" || arg == "-t") && i + 1 < argc){
            threshold = std::stof(argv[i++]);
        }else if ((arg == "--max_chunk_size" || arg == "-x") && i + 1 < argc){
            max_chunk_size = std::stoi(argv[i++]);
        }else if ((arg == "--min_chunk_size" || arg == "-n") && i + 1 < argc){
            min_chunk_size = std::stoi(argv[i++]);
        }else if ((arg == "--overlap" || arg == "-o") && i + 1 < argc){
            overlap = std::stoi(argv[i++]);
        }else if ((arg == "--path" || arg == "-p") && i + 1 < argc){
            output_fpath = argv[i++];
        }else if ((arg == "--help" || arg == "-h")){
            std::cout << help;
            return EXIT_SUCCESS;
        }else{
            std::cerr << "Unknown option: " << arg << " , use -h for options" << std::endl;
            return EXIT_FAILURE;
        }
    }


    std::ifstream inputFile(input_fpath);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file: " << input_fpath << std::endl;
        return EXIT_FAILURE;
    }
    std::string text((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());

    inputFile.close();

    if (text.empty()){
        std::cerr << "Input file is empty" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<Chunk> chunks = glue(text, threshold, max_chunk_size, min_chunk_size, overlap);

    if ((output_json(chunks, output_fpath) == -1)){
        std::cerr << "Failed to write output to file" << std::endl;
        return EXIT_FAILURE;
    }

}
