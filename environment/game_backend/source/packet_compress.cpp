#include "packet_compress.hpp"

void PacketCompress::extract() {
    tmpcount = 0;
    bitcount = 0;
    uint8_t counter = 0;
    uint8_t endy = 0;
    uint8_t temp = 0;
    int y;
    getBits(endy, 5);
    for (int c = 0; c < endy; c++) {
        for (int x = 0; x < 10; x++) square[21 - c][x] = 8;
        getBits(temp, 4);
        square[21 - c][temp] = 0;
    }
    for (int x = 0; x < 10; x++) {
        counter = 0;
        getBits(counter, 5);
        for (y = 0; y < counter; y++) square[y][x] = 0;
        for (; y < 22 - endy; y++) getBits(square[y][x], 3);
    }
    getBits(temp, 4);
    posX = temp - 2;
    getBits(temp, 5);
    posY = temp;
    getBits(piece, 3);
    getBits(color, 3);
    getBits(rotation, 2);
    getBits(nextpiece, 3);
    getBits(npcol, 3);
    getBits(nprot, 2);
    getBits(comboText, 5);
    getBits(pendingText, 8);
    getBits(bpmText, 8);
    getBits(comboTimerCount, 7);
    getBits(countdown, 2);
    getBigBits(time_val, 16);
}

void PacketCompress::getBits(uint8_t& byte, uint8_t bits) {
    byte = 0;
    uint8_t temp = 0;
    temp = m_data[tmpcount] >> bitcount | temp;
    bitcount += bits;
    if (bitcount > 7) {
        bitcount -= 8;
        tmpcount++;
        if (bitcount > 0) temp = m_data[tmpcount] << (bits - bitcount) | temp;
    }
    temp = temp << (8 - bits);
    temp = temp >> (8 - bits);
    byte = temp;
}

void PacketCompress::compress() {
    m_data.clear();
    tmpcount = 0;
    bitcount = 0;
    uint8_t counter = 0;
    int y, endy;
    for (endy = 21; endy >= 0; endy--) {
        if (square[endy][0] == 8 || square[endy][1] == 8)
            counter++;
        else
            break;
    }
    addBits(counter, 5);
    for (y = 21; y > endy; y--)
        for (uint8_t x = 0; x < 10; x++)
            if (square[y][x] == 0) {
                addBits(x, 4);
                break;
            }
    for (int x = 0; x < 10; x++) {
        counter = 0;
        for (y = 0; y <= endy; y++) {
            if (!square[y][x])
                counter++;
            else
                break;
        }
        addBits(counter, 5);
        for (; y <= endy; y++) {
            addBits(square[y][x], 3);
        }
    }
    uint8_t posx = 0, posy = 0;
    posx = posX + 2;
    posy = posY;
    addBits(posx, 4);
    addBits(posy, 5);
    addBits(piece, 3);
    addBits(color, 3);
    addBits(rotation, 2);
    addBits(nextpiece, 3);
    addBits(npcol, 3);
    addBits(npcol, 2);
    addBits(comboText, 5);
    addBits(pendingText, 8);
    addBits(bpmText, 8);  // make sure not bigger then 255
    // tmp = game->field.text.comboTimer.getPointCount() - 2;
    addBits(comboTimerCount, 7);
    // tmp = game->field.text.get<FieldText::Countdown>();
    addBits(countdown, 2);
    addBigBits(time_val, 16);
    /*if (!countdown) {
        uint16_t timevalue = game->gameclock.getElapsedTime().asMilliseconds();
        uint8_t smallpart = timevalue % 256;
        uint8_t bigpart = (timevalue - smallpart) / 256;
        addBits(smallpart, 8);
        addBits(bigpart, 8);
    }*/
}

void PacketCompress::addBits(uint8_t byte, uint8_t bits) {
    if (tmpcount >= m_data.size()) m_data.push_back(0);
    m_data[tmpcount] = m_data[tmpcount] | byte << bitcount;
    bitcount += bits;
    if (bitcount > 7) {
        bitcount -= 8;
        tmpcount++;
        if (bitcount > 0) {
            m_data.push_back(0);
            m_data[tmpcount] = m_data[tmpcount] | byte >> (bits - bitcount);
        }
    }
}

/*void PacketCompress::copy() {
    for (int y = 0; y < 22; y++)
        for (int x = 0; x < 10; x++) field->square[y][x] = square[y][x];
    field->piece.posX = posX;
    field->piece.posY = posY;
    field->piece.piece = piece;
    field->piece.tile = color;
    field->piece.rotation = rotation;
    field->updatePiece();
    field->nextpiece = nextpiece;
    field->npcol = npcol;
    field->nprot = nprot;
    if (!field->text.get<FieldText::Position>()) field->text.set<FieldText::BPM>(bpmText);
    field->text.set<FieldText::Pending>(pendingText);
    field->text.set<FieldText::Combo>(comboText);
    field->text.setComboTimer(comboTimerCount);
    if (countdown)
        field->text.set<FieldText::Countdown>(countdown);
    else
        field->text.hide<FieldText::Countdown>();
}*/

bool PacketCompress::validate() {
    for (int y = 0; y < 22; y++)
        for (int x = 0; x < 10; x++)
            if (square[y][x] > 8) return false;
    if (piece > 7) return false;
    if (color > 7) return false;
    if (rotation > 3) return false;
    if (posX > 8) return false;
    if (posY > 20) return false;
    if (nextpiece > 7) return false;
    if (npcol == 0 || npcol > 7) return false;
    if (nprot > 3) return false;
    if (countdown > 3) countdown = 0;

    return true;
}

void PacketCompress::loadTmp(const std::vector<uint8_t>& data) {
    tmpcount = 0;
    bitcount = 0;
    m_data = data;
}
