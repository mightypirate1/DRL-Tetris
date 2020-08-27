#ifndef PACKETCOMPRESS_H
#define PACKETCOMPRESS_H

#include <array>
#include <vector>

struct field_data {
    std::array<std::array<uint8_t, 10>, 22> square;
    int8_t posX, posY;
    uint8_t piece, color, rotation;
    uint8_t nextpiece, npcol, nprot;
    uint8_t comboText, pendingText, bpmText;
    uint8_t comboTimerCount;
    uint8_t countdown;
    uint16_t time_val;
};

class PacketCompress : public field_data {
   public:
    std::vector<uint8_t> m_data;

    void loadTmp(const std::vector<uint8_t>&);

    void compress();
    void extract();
    bool validate();

   protected:
    void addBits(uint8_t, uint8_t);
    void getBits(uint8_t&, uint8_t);

    template <class T>
    void addBigBits(T data, uint8_t bits) {
        uint8_t small_bits = 255;
        while (bits > 8) {
            addBits(data & small_bits, 8);
            bits -= 8;
            data = data >> 8;
        }
        addBits(static_cast<uint8_t>(data), bits);
    }

    template <class T>
    void getBigBits(T& data, uint8_t bits) {
        data = 0;
        uint8_t tmp, count = 0;
        while (bits > 8) {
            getBits(tmp, 8);
            data = data | (static_cast<T>(tmp) << (8 * count));
            bits -= 8;
            ++count;
        }
        getBits(tmp, bits);
        data = data | (static_cast<T>(tmp) << (8 * count));
    }

    uint8_t bitcount = 0;
    uint16_t tmpcount = 0;
};

#endif
