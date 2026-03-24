/// Sejong POS tagset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pos {
    NNG, NNP, NNB, NR, NP,
    VV, VA, VX, VCP, VCN,
    MAG, MAJ, MM,
    IC,
    JKS, JKC, JKG, JKO, JKB, JKV, JKQ, JX, JC,
    EP, EF, EC, ETN, ETM,
    XPN, XSN, XSV, XSA, XR,
    SF, SP, SS, SE, SO, SW, SH, SL, SN,
}

impl Pos {
    pub fn as_str(&self) -> &'static str {
        match self {
            Pos::NNG => "NNG", Pos::NNP => "NNP", Pos::NNB => "NNB",
            Pos::NR => "NR", Pos::NP => "NP",
            Pos::VV => "VV", Pos::VA => "VA", Pos::VX => "VX",
            Pos::VCP => "VCP", Pos::VCN => "VCN",
            Pos::MAG => "MAG", Pos::MAJ => "MAJ", Pos::MM => "MM",
            Pos::IC => "IC",
            Pos::JKS => "JKS", Pos::JKC => "JKC", Pos::JKG => "JKG",
            Pos::JKO => "JKO", Pos::JKB => "JKB", Pos::JKV => "JKV",
            Pos::JKQ => "JKQ", Pos::JX => "JX", Pos::JC => "JC",
            Pos::EP => "EP", Pos::EF => "EF", Pos::EC => "EC",
            Pos::ETN => "ETN", Pos::ETM => "ETM",
            Pos::XPN => "XPN", Pos::XSN => "XSN", Pos::XSV => "XSV",
            Pos::XSA => "XSA", Pos::XR => "XR",
            Pos::SF => "SF", Pos::SP => "SP", Pos::SS => "SS",
            Pos::SE => "SE", Pos::SO => "SO", Pos::SW => "SW",
            Pos::SH => "SH", Pos::SL => "SL", Pos::SN => "SN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub pos: Pos,
    pub start: usize,
    pub end: usize,
    pub score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct AnalyzeResult {
    pub tokens: Vec<Token>,
    pub score: f32,
    pub elapsed_ms: f64,
}
