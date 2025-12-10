import os
import boto3
from langchain_aws import ChatBedrock

# Initialize Bedrock clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# Your Knowledge Base IDs
VECTOR_KB_ID = "KXZ6KMQQUP"
GRAPH_KB_ID = "9GX3SMFMMH"

# Initialize Claude Haiku 3 via Bedrock
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    model_kwargs={
        "temperature": 0.0,
        "max_tokens": 4096
    }
)

def retrieve_from_kb(query, kb_id):
    """Retrieve documents from Bedrock Knowledge Base"""
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )
        return response.get('retrievalResults', [])
    except Exception as e:
        print(f"Error retrieving from KB {kb_id}: {e}")
        return []

def format_retrieval_results(results, source_type):
    """Format retrieval results into readable text"""
    if not results:
        return f"No results found from {source_type} knowledge base."
    
    formatted_text = f"\n--- {source_type} Knowledge Base Results ---\n"
    for idx, result in enumerate(results, 1):
        content = result.get('content', {}).get('text', '')
        score = result.get('score', 0.0)
        formatted_text += f"\n{idx}. (Score: {score:.3f})\n{content}\n"
    
    return formatted_text

def combine_and_generate_response(query, vector_results, graph_results, llm):
    """Combine results from both KBs and generate a unified response"""
    
    # Format results from both sources
    vector_context = format_retrieval_results(vector_results, "Vector")
    graph_context = format_retrieval_results(graph_results, "Graph")
    
    # Create a comprehensive prompt for the LLM
    synthesis_prompt = f"""You are an AI assistant tasked with answering questions using information from two different knowledge sources:
1. A vector-based knowledge base (semantic search results)
2. A graph-based knowledge base (relationship and entity-focused results)

User Question: {query}

{vector_context}

{graph_context}

Instructions:
- Synthesize the information from both knowledge bases
- Provide a comprehensive, coherent answer that combines the best insights from both sources
- If there are complementary details, integrate them seamlessly
- If there are contradictions, acknowledge them and provide the most accurate information
- Cite which source (Vector KB or Graph KB) specific information comes from when relevant
- Be concise but thorough

Provide your unified answer below:"""

    # Generate response using the LLM
    try:
        response = llm.invoke(synthesis_prompt)
        return response.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating unified response."

def hybrid_rag_query(query_text):
    """Main function to perform hybrid RAG across vector and graph KBs"""
    
    print(f"Processing query: {query_text}\n")
    print("="*80)
    
    # Step 1: Retrieve from Vector KB
    print("\n1. Retrieving from Vector Knowledge Base...")
    vector_results = retrieve_from_kb(query_text, VECTOR_KB_ID)
    print(f"   Retrieved {len(vector_results)} results from Vector KB")
    
    # Step 2: Retrieve from Graph KB
    print("\n2. Retrieving from Graph Knowledge Base...")
    graph_results = retrieve_from_kb(query_text, GRAPH_KB_ID)
    print(f"   Retrieved {len(graph_results)} results from Graph KB")
    
    # Step 3: Combine and synthesize
    print("\n3. Synthesizing unified response...")
    unified_response = combine_and_generate_response(
        query_text, 
        vector_results, 
        graph_results, 
        llm
    )
    
    return {
        'query': query_text,
        'vector_results_count': len(vector_results),
        'graph_results_count': len(graph_results),
        'unified_response': unified_response,
        'raw_vector_results': vector_results,
        'raw_graph_results': graph_results
    }

# Main execution
if __name__ == "__main__":
    try:
        query_text = "What are the differences between neptune databases and Neptune analytics"
        
        result = hybrid_rag_query(query_text)
        
        print("\n" + "="*80)
        print("UNIFIED RESPONSE:")
        print("="*80)
        print(result['unified_response'])
        print("\n" + "="*80)
        print(f"\nSummary:")
        print(f"  - Vector KB Results: {result['vector_results_count']}")
        print(f"  - Graph KB Results: {result['graph_results_count']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()