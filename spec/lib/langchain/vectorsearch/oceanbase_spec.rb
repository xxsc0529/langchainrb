# frozen_string_literal: true

require "sequel"
require "mysql2"

# Integration tests use the DB below. Override with OCEANBASE_URL env var.
# Ensure the database (e.g. langchain) exists: CREATE DATABASE IF NOT EXISTS langchain;
CONNECTION_ARGS = {
  host: "127.0.0.1",
  port: "2881",
  user: "root",
  password: "",
  db_name: "langchain"
}.freeze

def oceanbase_url
  ENV["OCEANBASE_URL"] || begin
    u = CONNECTION_ARGS[:user].to_s
    p = CONNECTION_ARGS[:password].to_s
    cred = p.empty? ? u : "#{u}:#{p}"
    "mysql2://#{cred}@#{CONNECTION_ARGS[:host]}:#{CONNECTION_ARGS[:port]}/#{CONNECTION_ARGS[:db_name]}"
  end
end

def oceanbase_available?
  db = Sequel.connect(oceanbase_url)
  db.test_connection
  db.disconnect
  true
rescue
  false
end

if oceanbase_available?
  # --- Integration tests against real OceanBase ---
  ob_url = oceanbase_url
  integration_index = "langchain_oceanbase_spec"

  RSpec.describe Langchain::Vectorsearch::Oceanbase, "integration" do
    let(:integration_llm) { double("llm", default_dimensions: 1536) }
    subject { described_class.new(url: ob_url, index_name: integration_index, llm: integration_llm, namespace: nil) }

    before do
      subject.create_default_schema
    end

    after do
      subject.destroy_default_schema
    end

    describe "#add_texts" do
      before do
        allow(integration_llm).to receive(:embed).with(text: anything).and_return(double(embedding: 1536.times.map { rand }))
      end

      it "adds texts" do
        result = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(result.size).to eq(2)
      end

      it "adds texts with metadata" do
        metadata = [
          {"source" => "doc1", "page" => 1},
          {"source" => "doc2", "page" => 2}
        ]
        result = subject.add_texts(
          texts: ["Hello World", "Hello World"],
          metadata: metadata
        )
        expect(result.size).to eq(2)
        db = Sequel.connect(ob_url)
        rows = db["SELECT id, metadata FROM `#{integration_index}` WHERE id IN (?, ?)", result[0], result[1]].all
        db.disconnect
        expect(rows.size).to eq(2)
        expect(JSON.parse(rows[0][:metadata])).to match(metadata[0])
        expect(JSON.parse(rows[1][:metadata])).to match(metadata[1])
      end
    end

    describe "#update_texts" do
      before do
        allow(integration_llm).to receive(:embed).with(text: anything).and_return(double(embedding: 1536.times.map { rand }))
      end

      it "updates texts" do
        ids = subject.add_texts(texts: ["Hello World", "Hello World"])
        result = subject.update_texts(
          texts: ["Hello World", "World Hello"],
          ids: ids
        )
        expect(result).to eq(ids)
      end
    end

    describe "#remove_texts" do
      before do
        allow(integration_llm).to receive(:embed).with(text: anything).and_return(double(embedding: 1536.times.map { rand }))
      end

      it "removes texts" do
        ids = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(ids.length).to eq(2)
        result = subject.remove_texts(ids: ids)
        expect(result).to eq(2)
      end
    end

    describe "#similarity_search and #similarity_search_by_vector" do
      embedding = 1536.times.map { 0 }

      before do
        allow(integration_llm).to receive(:embed).with(text: anything).and_return(double(embedding: 1536.times.map { rand }))
        allow(integration_llm).to receive(:embed).with(text: "earth").and_return(double(embedding: embedding))
        allow(integration_llm).to receive(:embed).with(text: "something about earth").and_return(double(embedding: embedding))
      end

      before do
        subject.add_texts(texts: ["something about earth"])
        subject.add_texts(texts: ["Hello World", "Hello World", "Hello World"])
      end

      it "similarity_search returns relevant document" do
        result = subject.similarity_search(query: "earth", k: 4)
        expect(result).to be_an(Array)
        expect(result.size).to be >= 1
        expect(result.first).to include(:content, :metadata)
        # With same embedding for "earth" and "something about earth", that doc should be in top results
        contents = result.map { |r| r[:content] }
        expect(contents).to include("something about earth")
      end

      it "similarity_search_by_vector returns results" do
        result = subject.similarity_search_by_vector(embedding: embedding, k: 3)
        expect(result).to be_an(Array)
        expect(result.size).to be >= 1
        expect(result.first).to include(:content, :metadata)
      end
    end
  end
end

# --- Unit tests (always run, with mocks) ---
RSpec.describe Langchain::Vectorsearch::Oceanbase do
  let(:index_name) { "documents" }
  let(:llm) { double("llm", default_dimensions: 1536) }
  let(:db) { double("db") }
  let(:table_ds) { double("table_ds") }

  subject do
    allow(Sequel).to receive(:connect).with("mysql2://user:pass@host:2881/db").and_return(db)
    described_class.new(
      url: "mysql2://user:pass@host:2881/db",
      index_name: index_name,
      llm: llm,
      namespace: nil
    )
  end

  before do
    allow(db).to receive(:[]).with(index_name.to_sym).and_return(table_ds)
  end

  describe ".format_vector" do
    it "formats embedding array as OceanBase VECTOR literal" do
      expect(described_class.format_vector([0.1, 0.2, 0.3])).to eq("[0.1,0.2,0.3]")
    end

    it "handles integer-like floats" do
      expect(described_class.format_vector([1.0, 2.0])).to eq("[1.0,2.0]")
    end
  end

  describe "#initialize" do
    it "sets table_name, namespace, vector_column and operator" do
      allow(Sequel).to receive(:connect).with("mysql2://u:p@h/d").and_return(db)
      vs = described_class.new(url: "mysql2://u:p@h/d", index_name: "t1", llm: llm)
      expect(vs.table_name).to eq("t1")
      expect(vs.namespace).to be_nil
      expect(vs.vector_column).to eq("vectors")
      expect(vs.operator).to eq("cosine_distance")
    end

    it "accepts distance_operator" do
      allow(Sequel).to receive(:connect).with("mysql2://u:p@h/d").and_return(db)
      vs = described_class.new(url: "mysql2://u:p@h/d", index_name: "t1", llm: llm, distance_operator: "l2_distance")
      expect(vs.operator).to eq("l2_distance")
    end
  end

  describe "#create_default_schema" do
    before do
      allow(db).to receive(:run)
      allow(llm).to receive(:default_dimensions).and_return(1536)
    end

    it "runs CREATE TABLE and CREATE VECTOR INDEX" do
      subject.create_default_schema
      expect(db).to have_received(:run).at_least(:twice)
    end

    context "when vector index already exists" do
      before do
        call_count = 0
        allow(db).to receive(:run) do
          call_count += 1
          raise Sequel::DatabaseError.new("Duplicate key name") if call_count > 1
        end
      end

      it "does not raise" do
        expect { subject.create_default_schema }.not_to raise_error
      end
    end
  end

  describe "#destroy_default_schema" do
    before do
      allow(db).to receive(:drop_table?).with(index_name.to_sym).and_return(nil)
    end

    it "drops the table" do
      subject.destroy_default_schema
      expect(db).to have_received(:drop_table?).with(index_name.to_sym)
    end
  end

  describe "#add_texts" do
    let(:embedding) { [0.1, 0.2, 0.3] }

    context "without ids" do
      before do
        allow(llm).to receive(:embed).with(text: "Hello").and_return(double(embedding: embedding))
        allow(table_ds).to receive(:insert).and_return(1, 2)
      end

      it "inserts and returns ids" do
        result = subject.add_texts(texts: ["Hello", "Hello"])
        expect(result).to eq([1, 2])
        expect(table_ds).to have_received(:insert).twice
      end
    end

    context "with ids" do
      before do
        allow(llm).to receive(:embed).with(text: "Hello").and_return(double(embedding: embedding))
        allow(table_ds).to receive(:replace)
      end

      it "replaces by id and returns ids" do
        result = subject.add_texts(texts: ["Hello"], ids: [10])
        expect(result).to eq([10])
        expect(table_ds).to have_received(:replace).with(hash_including(id: 10, content: "Hello"))
      end
    end
  end

  describe "#remove_texts" do
    before do
      allow(table_ds).to receive(:where).with(id: [1, 2]).and_return(table_ds)
      allow(table_ds).to receive(:delete).and_return(2)
    end

    it "deletes by ids" do
      subject.remove_texts(ids: [1, 2])
      expect(table_ds).to have_received(:delete)
    end
  end

  describe "#similarity_search_by_vector" do
    let(:embedding) { [0.1, 0.2, 0.3] }
    let(:fetch_ds) { double("fetch_dataset") }
    let(:select_ds) { double("select_ds") }

    before do
      allow(table_ds).to receive(:select).and_return(select_ds)
      allow(select_ds).to receive(:order).and_return(select_ds)
      allow(select_ds).to receive(:limit).and_return(select_ds)
      allow(select_ds).to receive(:where).and_return(select_ds)
      allow(select_ds).to receive(:sql).and_return("SELECT content, metadata, cosine_distance(vectors, '[0.1,0.2,0.3]') AS _dist FROM documents ORDER BY _dist LIMIT 2")
      allow(db).to receive(:fetch) do |sql|
        expect(sql).to include("cosine_distance")
        expect(sql).to include("APPROXIMATE LIMIT")
        fetch_ds
      end
      allow(fetch_ds).to receive(:all).and_return([
        {content: "Hello", metadata: "{}"},
        {content: "World", metadata: "{}"}
      ])
    end

    it "returns array of hashes with content and metadata" do
      result = subject.similarity_search_by_vector(embedding: embedding, k: 2)
      expect(result).to eq([
        {content: "Hello", metadata: "{}"},
        {content: "World", metadata: "{}"}
      ])
    end
  end

  describe "#similarity_search" do
    let(:embedding) { [0.1, 0.2, 0.3] }
    let(:fetch_ds) { double("fetch_dataset") }
    let(:select_ds) { double("select_ds") }

    before do
      allow(llm).to receive(:embed).with(text: "query").and_return(double(embedding: embedding))
      allow(table_ds).to receive(:select).and_return(select_ds)
      allow(select_ds).to receive(:order).and_return(select_ds)
      allow(select_ds).to receive(:limit).and_return(select_ds)
      allow(select_ds).to receive(:where).and_return(select_ds)
      allow(select_ds).to receive(:sql).and_return("SELECT content, metadata FROM documents LIMIT 1")
      allow(db).to receive(:fetch).and_return(fetch_ds)
      allow(fetch_ds).to receive(:all).and_return([{content: "Hi", metadata: nil}])
    end

    it "embeds query and calls similarity_search_by_vector" do
      result = subject.similarity_search(query: "query", k: 1)
      expect(result).to eq([{content: "Hi", metadata: nil}])
      expect(llm).to have_received(:embed).with(text: "query")
    end
  end
end
