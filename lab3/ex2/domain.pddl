(define (domain cleaning-robot)
  (:requirements :strips :typing)
  (:types robot room)

  (:predicates
    (at ?r - robot ?p - room)
    (dirty ?p - room)
    (clean ?p - room)
  )

  (:action move
    :parameters (?r - robot ?from ?to - room)
    :precondition (and (at ?r ?from) (not (= ?from ?to)))
    :effect (and (not (at ?r ?from)) (at ?r ?to))
  )

  (:action clean
    :parameters (?r - robot ?p - room)
    :precondition (and (at ?r ?p) (dirty ?p))
    :effect (and (not (dirty ?p)) (clean ?p))
  )
)