(define (problem clean-rooms)
  (:domain cleaning-robot)

  (:objects
    robo - robot
    room1 room2 room3 - room
  )

  (:init
    (at robo room1)
    (dirty room1)
    (dirty room2)
    (dirty room3)
  )

  (:goal (and
    (clean room1)
    (clean room2)
    (clean room3)
  ))
)