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

    pub fn from_str(s: &str) -> Option<Pos> {
        match s {
            "NNG" => Some(Pos::NNG), "NNP" => Some(Pos::NNP), "NNB" => Some(Pos::NNB),
            "NR" => Some(Pos::NR), "NP" => Some(Pos::NP),
            "VV" => Some(Pos::VV), "VA" => Some(Pos::VA), "VX" => Some(Pos::VX),
            "VCP" => Some(Pos::VCP), "VCN" => Some(Pos::VCN),
            "MAG" => Some(Pos::MAG), "MAJ" => Some(Pos::MAJ), "MM" => Some(Pos::MM),
            "IC" => Some(Pos::IC),
            "JKS" => Some(Pos::JKS), "JKC" => Some(Pos::JKC), "JKG" => Some(Pos::JKG),
            "JKO" => Some(Pos::JKO), "JKB" => Some(Pos::JKB), "JKV" => Some(Pos::JKV),
            "JKQ" => Some(Pos::JKQ), "JX" => Some(Pos::JX), "JC" => Some(Pos::JC),
            "EP" => Some(Pos::EP), "EF" => Some(Pos::EF), "EC" => Some(Pos::EC),
            "ETN" => Some(Pos::ETN), "ETM" => Some(Pos::ETM),
            "XPN" => Some(Pos::XPN), "XSN" => Some(Pos::XSN), "XSV" => Some(Pos::XSV),
            "XSA" => Some(Pos::XSA), "XR" => Some(Pos::XR),
            "SF" => Some(Pos::SF), "SP" => Some(Pos::SP), "SS" => Some(Pos::SS),
            "SE" => Some(Pos::SE), "SO" => Some(Pos::SO), "SW" => Some(Pos::SW),
            "SH" => Some(Pos::SH), "SL" => Some(Pos::SL), "SN" => Some(Pos::SN),
            _ => None,
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
