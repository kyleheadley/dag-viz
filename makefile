
all:
	cargo build --release
	ln -s target/release/dag_viz
run:
	cargo run --release

clean:
	cargo clean
	rm -f dag_viz