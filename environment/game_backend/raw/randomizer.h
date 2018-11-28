#ifndef RANDOMIZER_H
#define RANDOMIZER_H

#include <random>
#include <array>

template <typename T>
class UniformRealDistribution
{
 public:
    typedef T result_type;

 public:
    UniformRealDistribution(T _a = 0.0, T _b = 1.0)
        :m_a(_a),
         m_b(_b)
    {}

    void reset() {}

    template <class Generator>
    T operator()(Generator &_g)
    {
        double dScale = (m_b - m_a) / ((T)(_g.max() - _g.min()) + (T)1);
        return (_g() - _g.min()) * dScale  + m_a;
    }

    T a() const {return m_a;}
    T b() const {return m_b;}

 protected:
    T       m_a;
    T       m_b;
};

class randomizer {
public:
	randomizer();
	std::array<short, 7> noP;
	short total;

  std::array<float, 7> cogP;

  short lasthole;

  std::mt19937 hole_gen;
  UniformRealDistribution<float> hole_dist;

  std::mt19937 piece_gen;
  UniformRealDistribution<float> piece_dist;

  std::mt19937 AI_gen;

  short getHole(bool noStack=false);
  void seedHole(short seedNr);
  void seedPiece(short seedNr);
  short getPiece();
  double uniqueRnd();
  void reset();

  std::array<float, 7> getState();
  void setState(std::array<float, 7>);
};

#endif
