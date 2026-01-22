/*  
   Estrutura simples de dados com Rust
   Ordem de fila por nome
 */

use std::cmp::Ordering;
use std::fmt;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

#[derive(Eq)]
struct Pessoa {
    nome: String,
    idade: u32,
}   
impl Ord for Pessoa {
    fn cmp(&self, other: &Self) -> Ordering {
        self.nome.cmp(&other.nome)
    }
}
impl PartialOrd for Pessoa {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}   
impl PartialEq for Pessoa {
    fn eq(&self, other: &Self) -> bool {
        self.nome == other.nome
    }
}
impl Hash for Pessoa {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nome.hash(state);
    }
}
impl fmt::Display for Pessoa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({})", self.nome, self.idade)
    }
}   
fn main() {
    let mut fila: BinaryHeap<Pessoa> = BinaryHeap::new();

    fila.push(Pessoa { nome: String::from("Carlos"), idade: 30 });
    fila.push(Pessoa { nome: String::from("Ana"), idade: 25 });
    fila.push(Pessoa { nome: String::from("Bruno"), idade: 28 });

    while let Some(pessoa) = fila.pop() {
        println!("{}", pessoa);
    }
}