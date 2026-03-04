from src.signals.query_signals import extract_signals
from src.events.query_event import extract_event_from_query
from src.agents.agents import AnalystAgent, ContextAgent, WriterAgent

class RAGPipeline:
    def __init__(self, retriever, generator, graph):
        self.retriever = retriever
        self.generator = generator
        self.graph = graph

    def run(
        self,
        query: str,
        top_k: int = 5,
        signal_aware: bool = False,
        entity_aware: bool = False,
        graph_aware: bool = False
    ):
        # --- SIGNAL FILTER ---
        signals = None
        if signal_aware:
            signals = extract_signals(query)

        # --- GRAPH FILTER ---
        query_event = None
        affected_companies = None

        if graph_aware and self.graph:
            query_event = extract_event_from_query(query)
            if query_event:
                affected = self.graph.query_by_event_type(query_event)
                affected_companies = {e["company"] for e in affected}
                print("COMPANIES:", affected_companies)

        # --- FINAL FILTER ---
        def active_filter(doc):

            # Signal constraint
            if signals:
                if signals.growth_direction == "positive" and doc["growth"] <= 0:
                    return False
                if signals.growth_direction == "negative" and doc["growth"] >= 0:
                    return False
                if signals.sector and doc["sector"] != signals.sector:
                    return False

            # Graph constraint
            if affected_companies is not None:
                if doc["company"] not in affected_companies:
                    return False

            return True

        # --- RETRIEVAL ---
        if signal_aware or graph_aware:
            if entity_aware:
                docs = self.retriever.retrieve_entity_aware(
                    query,
                    top_k=top_k,
                    filter_fn=active_filter
                )
            else:
                docs = self.retriever.retrieve_with_filter(
                    query,
                    top_k=top_k,
                    filter_fn=active_filter
                )

            if not docs and not graph_aware:
                docs = self.retriever.retrieve(query, top_k)
        else:
            docs = self.retriever.retrieve(query, top_k)

        # --- MULTI-AGENT REASONING ---

        analyst = AnalystAgent()
        context_agent = ContextAgent()
        writer = WriterAgent(self.generator)

        analysis = analyst.analyze(docs)
        context_info = context_agent.summarize_context(docs)

        answer = writer.write(query, analysis, context_info)

        return {
            "query": query,
            "answer": answer,
            "documents": docs,
            "analysis": analysis,
            "context": context_info,
        }