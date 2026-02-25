# frozen_string_literal: true

module Langchain::Vectorsearch
  #
  # OceanBase vector search adapter (aligned with pyobvector, MySQL protocol compatible).
  #
  # Gem requirements:
  #     gem "sequel", "~> 5.87.0"
  #     gem "mysql2", "~> 0.5"
  #
  # Usage:
  #     oceanbase = Langchain::Vectorsearch::Oceanbase.new(
  #       url: "mysql2://user:password@host:2881/database",
  #       index_name: "documents",
  #       llm: llm,
  #       namespace: nil
  #     )
  #
  class Oceanbase < Base
    OPERATORS = {
      "cosine_distance" => "cosine_distance",
      "l2_distance" => "l2_distance",
      "inner_product" => "inner_product",
      "negative_inner_product" => "negative_inner_product"
    }
    DEFAULT_OPERATOR = "cosine_distance"

    attr_reader :db, :operator, :table_name, :namespace_column, :namespace, :vector_column

    # @param url [String] MySQL protocol connection URL, e.g. mysql2://user:password@host:2881/dbname
    # @param index_name [String] Table name (used as collection/index)
    # @param llm [Object] LLM used to generate embeddings
    # @param namespace [String, nil] Namespace for multi-tenant filtering
    # @param distance_operator [String] Distance function: cosine_distance / l2_distance / inner_product / negative_inner_product
    def initialize(url:, index_name:, llm:, namespace: nil, distance_operator: DEFAULT_OPERATOR)
      depends_on "sequel"
      depends_on "mysql2"

      @db = Sequel.connect(url)
      @table_name = index_name
      @namespace_column = "namespace"
      @namespace = namespace
      @vector_column = "vectors"
      @operator = OPERATORS[distance_operator] || OPERATORS[DEFAULT_OPERATOR]

      super(llm: llm)
    end

    # Format embedding array as OceanBase VECTOR literal.
    # @param embedding [Array<Float>]
    # @return [String] e.g. "[0.1,0.2,0.3]"
    def self.format_vector(embedding)
      "[#{embedding.map { |v| Float(v) }.join(",")}]"
    end

    def format_vector(embedding)
      self.class.format_vector(embedding)
    end

    # Batch upsert: update if exists, insert otherwise (MySQL/OceanBase REPLACE INTO, same as pyobvector ReplaceStmt).
    def upsert_texts(texts:, ids:, metadata: nil)
      metadata = Array.new(texts.size, {}) if metadata.nil?

      texts.zip(ids, metadata).each do |text, id, meta|
        vec_str = format_vector(llm.embed(text: text).embedding)
        db[table_name.to_sym].replace(
          :id => id,
          :content => text,
          vector_column.to_sym => vec_str,
          namespace_column.to_sym => namespace,
          :metadata => meta.to_json
        )
      end
      ids
    end

    def add_texts(texts:, ids: nil, metadata: nil)
      metadata = Array.new(texts.size, {}) if metadata.nil?

      if ids.nil? || ids.empty?
        inserted = []
        texts.zip(metadata).each do |text, meta|
          vec_str = format_vector(llm.embed(text: text).embedding)
          row = {
            :content => text,
            vector_column.to_sym => vec_str,
            namespace_column.to_sym => namespace,
            :metadata => meta.to_json
          }
          id = db[table_name.to_sym].insert(row)
          inserted << id
        end
        inserted
      else
        upsert_texts(texts: texts, ids: ids, metadata: metadata)
      end
    end

    def update_texts(texts:, ids:, metadata: nil)
      upsert_texts(texts: texts, ids: ids, metadata: metadata)
    end

    def remove_texts(ids:)
      db[table_name.to_sym].where(id: ids).delete
    end

    # Maps distance function name to OceanBase vector index distance parameter.
    INDEX_DISTANCE_PARAM = {
      "cosine_distance" => "cosine",
      "l2_distance" => "l2",
      "inner_product" => "inner_product",
      "negative_inner_product" => "negative_inner_product"
    }.freeze

    def create_default_schema
      dim = llm.default_dimensions
      # OceanBase VECTOR type (see pyobvector).
      db.run <<~SQL
        CREATE TABLE IF NOT EXISTS `#{table_name}` (
          id BIGINT PRIMARY KEY AUTO_INCREMENT,
          content TEXT,
          #{vector_column} VECTOR(#{dim}),
          #{namespace_column} VARCHAR(255) DEFAULT NULL,
          metadata JSON DEFAULT NULL
        )
      SQL
      # Create HNSW vector index for approximate nearest neighbor (OceanBase docs: distance=l2, type=hnsw, lib=vsag).
      index_name_sql = "idx_#{table_name}_#{vector_column}"
      distance_param = INDEX_DISTANCE_PARAM[operator] || "cosine"
      db.run "CREATE VECTOR INDEX `#{index_name_sql}` ON `#{table_name}` (#{vector_column}) WITH (distance=#{distance_param}, type=hnsw)"
    rescue Sequel::DatabaseError => e
      raise unless e.message.match?(/Duplicate key name|already exists|1061/)
      # Ignore if vector index already exists.
    end

    def destroy_default_schema
      db.drop_table?(table_name.to_sym)
    end

    def similarity_search(query:, k: 4)
      embedding = llm.embed(text: query).embedding
      similarity_search_by_vector(embedding: embedding, k: k)
    end

    # ANN search using OceanBase vector distance functions + APPROXIMATE (see pyobvector ann_search).
    # @return [Array<Hash>] Hashes with :content, :metadata, etc.
    def similarity_search_by_vector(embedding:, k: 4)
      vec_str = format_vector(embedding)
      vec_escaped = vec_str.gsub("'", "''")
      dist_expr = "#{operator}(#{vector_column}, '#{vec_escaped}')"
      ns_col = namespace_column.to_sym

      ds = db[table_name.to_sym]
        .select(Sequel[:content], Sequel[:metadata], Sequel.lit("#{dist_expr} AS _dist"))
        .order(Sequel.lit(dist_expr))
        .limit(k)

      ds = ds.where(ns_col => namespace) if namespace

      # OceanBase approximate nearest neighbor: append APPROXIMATE limit k to SQL.
      sql = ds.sql
      sql = sql.sub(/\s+LIMIT\s+\d+\s*$/i) { " APPROXIMATE LIMIT #{k}" }

      rows = db.fetch(sql).all
      rows.map { |r| {content: r[:content], metadata: r[:metadata]} }
    end

    def ask(question:, k: 4, &block)
      search_results = similarity_search(query: question, k: k)
      context = search_results.map { |r| r[:content].to_s }.join("\n---\n")
      prompt = generate_rag_prompt(question: question, context: context)
      messages = [{role: "user", content: prompt}]
      response = llm.chat(messages: messages, &block)
      response.context = context
      response
    end
  end
end
