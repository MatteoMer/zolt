const std = @import("std");

pub fn cleanLine(raw_line: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, raw_line, " \t\r");
    if (trimmed.len == 0 or trimmed[0] == '#') return null;
    return trimmed;
}

pub fn splitFieldsExact(comptime field_count: usize, line: []const u8, separator: u8) ![field_count][]const u8 {
    var parts = std.mem.splitScalar(u8, line, separator);
    var fields: [field_count][]const u8 = undefined;

    for (0..field_count) |i| {
        fields[i] = std.mem.trim(u8, parts.next() orelse return error.NotEnoughFields, " \t\r");
    }

    if (parts.next() != null) return error.TooManyFields;
    return fields;
}

pub fn parseDecimal(comptime T: type, text: []const u8) !T {
    return std.fmt.parseInt(T, std.mem.trim(u8, text, " \t\r"), 10);
}

pub fn parseHexBytesExact(comptime byte_len: usize, text: []const u8) ![byte_len]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r");
    if (trimmed.len != byte_len * 2) return error.InvalidHexLength;

    var out: [byte_len]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, trimmed);
    return out;
}

pub fn parseCsvExact(comptime T: type, text: []const u8, allocator: std.mem.Allocator, parse_one: fn ([]const u8) anyerror!T) ![]T {
    var parts = std.mem.splitScalar(u8, std.mem.trim(u8, text, " \t\r"), ',');
    var values: std.ArrayListUnmanaged(T) = .empty;
    errdefer values.deinit(allocator);

    while (parts.next()) |raw_part| {
        const part = std.mem.trim(u8, raw_part, " \t\r");
        if (part.len == 0) continue;
        try values.append(allocator, try parse_one(part));
    }

    return values.toOwnedSlice(allocator);
}
