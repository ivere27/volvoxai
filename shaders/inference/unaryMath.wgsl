@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
struct Params { size : u32, operation : u32, pad0 : u32, pad1 : u32 }
@group(0) @binding(2) var<uniform> params : Params;

// Split F32 representations of 2^k mod 2pi, k=0..127. Each pair was
// rounded from a 180-decimal-digit mathematical remainder, independently;
// repeated doubling at F32 precision would accumulate phase error.
const POW2_MOD_TAU = array<vec2<f32>, 128>(
    vec2<f32>(1, 0),
    vec2<f32>(2, 0),
    vec2<f32>(4, 0),
    vec2<f32>(1.7168146371841431, 5.5636270701597823e-08),
    vec2<f32>(3.4336292743682861, 1.1127254140319565e-07),
    vec2<f32>(0.58407348394393921, -1.9841872145320849e-08),
    vec2<f32>(1.1681469678878784, -3.9683744290641698e-08),
    vec2<f32>(2.3362939357757568, -7.9367488581283396e-08),
    vec2<f32>(4.6725878715515137, -1.5873497716256679e-07),
    vec2<f32>(3.0619900226593018, 9.5794192134235345e-08),
    vec2<f32>(6.1239800453186035, 1.9158838426847069e-07),
    vec2<f32>(5.9647750854492188, 8.1185177691622812e-08),
    vec2<f32>(5.6463651657104492, -1.3962124967292766e-07),
    vec2<f32>(5.009544849395752, -1.0439693198804889e-07),
    vec2<f32>(3.7359042167663574, -3.3948303723718709e-08),
    vec2<f32>(1.1886230707168579, -1.2260334969482756e-08),
    vec2<f32>(2.3772461414337158, -2.4520669938965511e-08),
    vec2<f32>(4.7544922828674316, -4.9041339877931023e-08),
    vec2<f32>(3.2257990837097168, 7.6762880496517027e-08),
    vec2<f32>(0.16841301321983337, 5.4577253827403638e-10),
    vec2<f32>(0.33682602643966675, 1.0915450765480728e-09),
    vec2<f32>(0.6736520528793335, 2.1830901530961455e-09),
    vec2<f32>(1.347304105758667, 4.366180306192291e-09),
    vec2<f32>(2.694608211517334, 8.7323606123845821e-09),
    vec2<f32>(5.389216423034668, 1.7464721224769164e-08),
    vec2<f32>(4.4952473640441895, 2.0977500980734476e-07),
    vec2<f32>(2.7073097229003906, 1.1755840745308888e-07),
    vec2<f32>(5.4146194458007812, 2.3511681490617775e-07),
    vec2<f32>(4.5460538864135742, 1.6824203896703693e-07),
    vec2<f32>(2.8089227676391602, 3.4492469325186903e-08),
    vec2<f32>(5.6178455352783203, 6.8984938650373806e-08),
    vec2<f32>(4.9525060653686523, -1.6402171354457096e-07),
    vec2<f32>(3.6218264102935791, 8.5220705159372301e-08),
    vec2<f32>(0.96046769618988037, -1.2340896304863236e-08),
    vec2<f32>(1.9209353923797607, -2.4681792609726472e-08),
    vec2<f32>(3.8418707847595215, -4.9363585219452943e-08),
    vec2<f32>(1.400556206703186, -4.3090899737308064e-08),
    vec2<f32>(2.8011124134063721, -8.6181799474616128e-08),
    vec2<f32>(5.6022248268127441, -1.7236359894923226e-07),
    vec2<f32>(4.9212641716003418, -1.6988164475151279e-07),
    vec2<f32>(3.559342622756958, 7.3500842745488626e-08),
    vec2<f32>(0.83550006151199341, 2.3824016537332682e-08),
    vec2<f32>(1.6710001230239868, 4.7648033074665364e-08),
    vec2<f32>(3.3420002460479736, 9.5296066149330727e-08),
    vec2<f32>(0.40081536769866943, 7.8098256750536166e-09),
    vec2<f32>(0.80163073539733887, 1.5619651350107233e-08),
    vec2<f32>(1.6032614707946777, 3.1239302700214466e-08),
    vec2<f32>(3.2065229415893555, 6.2478605400428933e-08),
    vec2<f32>(0.12986069917678833, 1.7795410700571779e-09),
    vec2<f32>(0.25972139835357666, 3.5590821401143558e-09),
    vec2<f32>(0.51944279670715332, 7.1181642802287115e-09),
    vec2<f32>(1.0388855934143066, 1.4236328560457423e-08),
    vec2<f32>(2.0777711868286133, 2.8472657120914846e-08),
    vec2<f32>(4.1555423736572266, 5.6945314241829692e-08),
    vec2<f32>(2.0278995037078857, 5.0317606081762278e-08),
    vec2<f32>(4.0557990074157715, 1.0063521216352456e-07),
    vec2<f32>(1.8284128904342651, 1.8488123032511794e-08),
    vec2<f32>(3.6568257808685303, 3.6976246065023588e-08),
    vec2<f32>(1.0304663181304932, 1.037947150450691e-08),
    vec2<f32>(2.0609326362609863, 2.0758943009013819e-08),
    vec2<f32>(4.1218652725219727, 4.1517886018027639e-08),
    vec2<f32>(1.9605453014373779, 1.946275496322869e-08),
    vec2<f32>(3.9210906028747559, 3.892550992645738e-08),
    vec2<f32>(1.5589959621429443, 1.4277998339196074e-08),
    vec2<f32>(3.1179919242858887, 2.8555996678392148e-08),
    vec2<f32>(6.2359838485717773, 5.7111993356784296e-08),
    vec2<f32>(6.1887826919555664, -1.8776761123717733e-07),
    vec2<f32>(6.0943799018859863, -2.0068966932740295e-07),
    vec2<f32>(5.9055743217468262, -2.2653377129699948e-07),
    vec2<f32>(5.5279626846313477, 1.9861518296693248e-07),
    vec2<f32>(4.772740364074707, 9.523877508854639e-08),
    vec2<f32>(3.2622957229614258, -1.1151404066822579e-07),
    vec2<f32>(0.2414059191942215, -3.4790423875108445e-09),
    vec2<f32>(0.48281183838844299, -6.958084775021689e-09),
    vec2<f32>(0.96562367677688599, -1.3916169550043378e-08),
    vec2<f32>(1.931247353553772, -2.7832339100086756e-08),
    vec2<f32>(3.8624947071075439, -5.5664678200173512e-08),
    vec2<f32>(1.441804051399231, -5.5693089251462879e-08),
    vec2<f32>(2.8836081027984619, -1.1138617850292576e-07),
    vec2<f32>(5.7672162055969238, -2.2277235700585152e-07),
    vec2<f32>(5.251246452331543, 2.0613801154922839e-07),
    vec2<f32>(4.2193078994750977, 1.1028442514771086e-07),
    vec2<f32>(2.155430793762207, -8.1422754760751559e-08),
    vec2<f32>(4.3108615875244141, -1.6284550952150312e-07),
    vec2<f32>(2.3385374546051025, 8.7573134521790053e-08),
    vec2<f32>(4.6770749092102051, 1.7514626904358011e-07),
    vec2<f32>(3.0709648132324219, 4.8300929478273247e-08),
    vec2<f32>(6.1419296264648438, 9.6601858956546494e-08),
    vec2<f32>(6.0006742477416992, -1.0878787293222558e-07),
    vec2<f32>(5.718163013458252, -4.273018916478577e-08),
    vec2<f32>(5.1531405448913574, 8.9385174817380175e-08),
    vec2<f32>(4.0230960845947266, -1.2322124121055822e-07),
    vec2<f32>(1.7630065679550171, 4.7612367382043885e-08),
    vec2<f32>(3.5260131359100342, 9.5224734764087771e-08),
    vec2<f32>(0.76884114742279053, 7.6671549109619264e-09),
    vec2<f32>(1.5376822948455811, 1.5334309821923853e-08),
    vec2<f32>(3.0753645896911621, 3.0668619643847705e-08),
    vec2<f32>(6.1507291793823242, 6.1337239287695411e-08),
    vec2<f32>(6.0182733535766602, -1.7931711226992775e-07),
    vec2<f32>(5.7533612251281738, -1.8378867139290378e-07),
    vec2<f32>(5.2235369682312012, -1.9273177542800113e-07),
    vec2<f32>(4.1638884544372559, -2.1061798349819583e-07),
    vec2<f32>(2.0445911884307861, -7.9718294188069194e-09),
    vec2<f32>(4.0891823768615723, -1.5943658837613839e-08),
    vec2<f32>(1.8951793909072876, 2.3748953026370145e-08),
    vec2<f32>(3.7903587818145752, 4.7497906052740291e-08),
    vec2<f32>(1.297532320022583, 3.1422796809010833e-08),
    vec2<f32>(2.595064640045166, 6.2845593618021667e-08),
    vec2<f32>(5.190129280090332, 1.2569118723604333e-07),
    vec2<f32>(4.0970735549926758, -5.0609237689513975e-08),
    vec2<f32>(1.9109617471694946, -4.5582204677430127e-08),
    vec2<f32>(3.8219234943389893, -9.1164409354860254e-08),
    vec2<f32>(1.360661506652832, -7.4832549046277563e-09),
    vec2<f32>(2.7213230133056641, -1.4966509809255513e-08),
    vec2<f32>(5.4426460266113281, -2.9933019618511025e-08),
    vec2<f32>(4.6021065711975098, 1.1497952101535702e-07),
    vec2<f32>(2.9210281372070312, -7.203256302545924e-08),
    vec2<f32>(5.8420562744140625, -1.4406512605091848e-07),
    vec2<f32>(5.4009270668029785, -1.1328468474403053e-07),
    vec2<f32>(4.5186686515808105, -5.1723805682968305e-08),
    vec2<f32>(2.7541518211364746, 7.1397948886442464e-08),
    vec2<f32>(5.5083036422729492, 1.4279589777288493e-07),
    vec2<f32>(4.7334222793579102, -1.6399802404976072e-08),
    vec2<f32>(3.1836593151092529, -9.637262365913557e-08),
    vec2<f32>(0.084133133292198181, -2.9985240956875714e-09),
    vec2<f32>(0.16826626658439636, -5.9970481913751428e-09),
    vec2<f32>(0.33653253316879272, -1.1994096382750286e-08),
    vec2<f32>(0.67306506633758545, -2.3988192765500571e-08),
);

fn add_split(a : vec2<f32>, b : vec2<f32>) -> vec2<f32> {
    let sum = a.x + b.x;
    let virtual_value = sum - a.x;
    let residual = (a.x - (sum - virtual_value)) + (b.x - virtual_value) + a.y + b.y;
    let high = sum + residual;
    return vec2<f32>(high, residual - (high - sum));
}

// Break the exact F32 value into at most 24 powers of two and reduce with the
// immutable split remainders; sin/cos receive an argument inside [-pi,pi].
fn reduced_angle(angle : f32) -> f32 {
    if (angle <= 3.1415927410125732421875) { return angle; }
    let bits = bitcast<u32>(angle);
    let exponent = i32((bits >> 23u) & 255u) - 127;
    let mantissa = (bits & 8388607u) | 8388608u;
    let tau = vec2<f32>(6.283185482025146484375, -0.0000001748455531469517);
    var remainder = vec2<f32>(0.0, 0.0);
    for (var bit = 0u; bit < 24u; bit = bit + 1u) {
        if ((mantissa & (1u << bit)) == 0u) { continue; }
        let power = exponent - 23 + i32(bit);
        var part : vec2<f32>;
        if (power >= 0) {
            part = POW2_MOD_TAU[u32(power)];
        } else {
            part = vec2<f32>(bitcast<f32>(u32(127 + power) << 23u), 0.0);
        }
        remainder = add_split(remainder, part);
        if (remainder.x > tau.x || (remainder.x == tau.x && remainder.y >= tau.y)) {
            remainder = add_split(remainder, -tau);
        }
    }
    if (remainder.x + remainder.y > 3.1415927410125732421875) {
        remainder = add_split(remainder, -tau);
    }
    return remainder.x + remainder.y;
}

// WGSL built-in trigonometry permits comparatively loose approximation error.
// On the reduced interval these polynomials keep rotation error near F32
// rounding on both hardware and software implementations.
fn rotation(angle : f32) -> vec2<f32> {
    var x = angle;
    var cosine_sign = 1.0;
    if (x > 1.57079637050628662109375) {
        x = 3.1415927410125732421875 - x;
        cosine_sign = -1.0;
    } else if (x < -1.57079637050628662109375) {
        x = -3.1415927410125732421875 - x;
        cosine_sign = -1.0;
    }
    let square = x * x;
    let sine = x * (1.0 + square * (-0.1666666666666667 + square * (0.008333333333333333 +
        square * (-0.0001984126984126984 + square * (0.0000027557319223985893 +
        square * (-0.00000002505210838544172 + square * 0.00000000016059043836821615))))));
    let cosine = 1.0 + square * (-0.5 + square * (0.041666666666666664 +
        square * (-0.001388888888888889 + square * (0.0000248015873015873 +
        square * (-0.0000002755731922398589 + square * 0.00000000208767569878681)))));
    return vec2<f32>(sine, cosine * cosine_sign);
}

// Stable log(1+z), 0 <= z <= 1, also preserving tiny Softplus tails.
fn log1p_positive(z : f32) -> f32 {
    if (z < 0.0001) { return z * (1.0 - z * 0.5 + z * z / 3.0); }
    let r = z / (2.0 + z); let q = r * r;
    return 2.0 * r * (1.0 + q * (1.0/3.0 + q * (1.0/5.0 + q *
        (1.0/7.0 + q * (1.0/9.0 + q * (1.0/11.0 + q * (1.0/13.0 + q/15.0)))))));
}
fn atan_reduced(value : f32) -> f32 {
    let bits = bitcast<u32>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u || value == 0.0) { return value; }
    var x = abs(value); var offset = 0.0;
    let reciprocal = x > 1.0;
    if (reciprocal) { x = 1.0 / x; }
    if (x > 0.414213562373095) { x = (x - 1.0) / (x + 1.0); offset = 0.7853981633974483; }
    let q = x * x;
    var result = offset + x * (1.0 + q * (-1.0/3.0 + q * (1.0/5.0 + q *
        (-1.0/7.0 + q * (1.0/9.0 + q * (-1.0/11.0 + q * (1.0/13.0 + q *
        (-1.0/15.0 + q * (1.0/17.0 - q/19.0)))))))));
    if (reciprocal) { result = 1.5707963267948966 - result; }
    return bitcast<f32>(bitcast<u32>(result) | (bits & 0x80000000u));
}
fn round_even(value : f32) -> f32 {
    if (!(abs(value) < 8388608.0)) { return value; }
    var base = floor(abs(value)); let fraction = abs(value) - base;
    if (fraction > 0.5 || (fraction == 0.5 && (u32(base) & 1u) != 0u)) { base += 1.0; }
    return bitcast<f32>(bitcast<u32>(base) | (bitcast<u32>(value) & 0x80000000u));
}
fn erf_value(value : f32) -> f32 {
    if (value == 0.0) { return value; }
    let x = abs(value); let t = 1.0 / (1.0 + 0.3275911 * x);
    let p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
        0.284496736) * t + 0.254829592) * t);
    let result = 1.0 - p * exp(-x*x);
    return bitcast<f32>(bitcast<u32>(result) | (bitcast<u32>(value) & 0x80000000u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) grid : vec3<u32>) {
    let row_start = gid.y * (grid.x * 64u);
    if (row_start >= params.size || gid.x >= params.size - row_start) { return; }
    let index = row_start + gid.x;
    let value = input[index];
    var result = 0.0;
    if (params.operation == 0u) {
        // Preserve NaN and infinities explicitly; finite Exp is never clamped.
        let bits = bitcast<u32>(value);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            result = select(value, 0.0, bits == 0xff800000u);
        } else {
            result = exp(value);
        }
    } else if (params.operation <= 2u) {
        let bits = bitcast<u32>(value);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            result = bitcast<f32>(bits | 0x00400000u);
        } else {
            let angle = reduced_angle(abs(value));
            let pair = rotation(angle);
            result = select(pair.y, select(pair.x, -pair.x, (bits >> 31u) != 0u),
                            params.operation == 1u);
        }
    }
    if (params.operation == 3u) { result = sqrt(value); }
    else if (params.operation == 4u) {
        let bits = bitcast<u32>(value); let magnitude = bits & 0x7fffffffu;
        if (magnitude > 0x7f800000u || value < 0.0) { result = bitcast<f32>(bits | 0x7fc00000u); }
        else if (magnitude == 0u) { result = bitcast<f32>(bits | 0xff800000u); }
        else if (magnitude == 0x7f800000u) { result = value; }
        else { result = log(value); }
    }
    else if (params.operation == 5u) { result = erf_value(value); }
    else if (params.operation == 6u) { result = floor(value); }
    else if (params.operation == 7u) { result = ceil(value); }
    else if (params.operation == 8u) { result = round_even(value); }
    else if (params.operation == 9u) { result = atan_reduced(value); }
    else if (params.operation == 10u) { result = max(value, 0.0) + log1p_positive(exp(-abs(value))); }
    else if (params.operation == 11u) { result = -value; }
    output[index] = result;
}
