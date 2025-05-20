(define (domain package-transport)
  (:requirements :strips :typing :negative-preconditions :action-costs)
  (:types location vehicle package)
  (:predicates
    (vehicle-at ?v - vehicle ?l - location)  ; Changed from 'at'
    (at ?p - package ?l - location)
    (in ?p - package ?v - vehicle)
    (connected ?from ?to - location)
  )
  (:functions (distance ?from ?to - location) (fuel-cost))
  (:action load
    :parameters (?p - package ?v - vehicle ?l - location)
    :precondition (and (at ?p ?l) (vehicle-at ?v ?l) (not (in ?p ?v)))  ; Updated here
    :effect (and (in ?p ?v) (not (at ?p ?l))))
  (:action unload
    :parameters (?p - package ?v - vehicle ?l - location)
    :precondition (and (in ?p ?v) (vehicle-at ?v ?l))
    :effect (and (at ?p ?l) (not (in ?p ?v))))
  (:action drive
    :parameters (?v - vehicle ?from ?to - location)
    :precondition (and (vehicle-at ?v ?from) (connected ?from ?to))  ; Updated here
    :effect (and (vehicle-at ?v ?to) (not (vehicle-at ?v ?from))  ; Updated here
              (increase (fuel-cost) (distance ?from ?to))))
)