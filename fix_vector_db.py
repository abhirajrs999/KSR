import sys
sys.path.append('src')

from database.vector_store import ChromaVectorStore

# Clear the existing collection
vector_store = ChromaVectorStore()
try:
    collection = vector_store.get_collection()
    all_data = collection.get()
    if all_data['ids']:
        collection.delete(ids=all_data['ids'])
        print(f'✅ Cleared {len(all_data["ids"])} old documents')
    else:
        print('✅ Collection was already empty')
    
    print('Vector database cleared. Please run 03_create_vector_db.py again.')
    
except Exception as e:
    print(f'❌ Error: {e}')
