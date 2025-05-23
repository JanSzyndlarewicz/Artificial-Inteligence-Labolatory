(define (domain paczki)
  (:requirements 
    :strips
    :typing
    :negative-preconditions
    :durative-actions
    :action-costs
    :equality)
  
  (:types
    paczka lokacja pojazd - object
    samochod samolot lodz - pojazd
    typ_transportu - object
  )
  
  (:constants
    drogowe lotnicze wodne - typ_transportu
  )
  
  (:predicates
    (at ?x - (either paczka pojazd) ?l - lokacja)
    (w ?p - paczka ?v - pojazd)
    (wolny ?v - pojazd)
    (dostarczona ?p - paczka)
    (polaczone ?l1 ?l2 - lokacja ?typ - typ_transportu)
  )
  
  (:functions
    (koszt-drogowy ?l1 ?l2 - lokacja)
    (koszt-lotniczy ?l1 ?l2 - lokacja)
    (koszt-wodny ?l1 ?l2 - lokacja)
    (czas-drogowy ?l1 ?l2 - lokacja)
    (czas-lotniczy ?l1 ?l2 - lokacja)
    (czas-wodny ?l1 ?l2 - lokacja)
    (total-cost)
  )
  
  ;; Loading action
  (:durative-action zaladuj
    :parameters (?p - paczka ?v - pojazd ?l - lokacja)
    :duration (= ?duration 1)
    :condition (and
      (at start (at ?p ?l))
      (at start (at ?v ?l))
      (at start (wolny ?v)))
    :effect (and
      (at start (not (at ?p ?l)))
      (at start (not (wolny ?v)))
      (at end (w ?p ?v))
      (at start (increase (total-cost) 10)))
  )

  ;; Unloading action
  (:durative-action rozladuj
    :parameters (?p - paczka ?v - pojazd ?l - lokacja)
    :duration (= ?duration 1)
    :condition (and
      (at start (w ?p ?v))
      (at start (at ?v ?l)))
    :effect (and
      (at start (not (w ?p ?v)))
      (at end (at ?p ?l))
      (at end (wolny ?v))
      (at end (dostarczona ?p))
      (at start (increase (total-cost) 10)))
  )

  ;; Road transport
  (:durative-action jedz
    :parameters (?v - samochod ?from ?to - lokacja)
    :duration (= ?duration (czas-drogowy ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (polaczone ?from ?to drogowe)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (koszt-drogowy ?from ?to))))
  )

  ;; Air transport
  (:durative-action lec
    :parameters (?v - samolot ?from ?to - lokacja)
    :duration (= ?duration (czas-lotniczy ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (polaczone ?from ?to lotnicze)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (koszt-lotniczy ?from ?to))))
  )

  ;; Water transport
  (:durative-action plyn
    :parameters (?v - lodz ?from ?to - lokacja)
    :duration (= ?duration (czas-wodny ?from ?to))
    :condition (and
      (at start (at ?v ?from))
      (over all (polaczone ?from ?to wodne)))
    :effect (and
      (at start (not (at ?v ?from)))
      (at end (at ?v ?to))
      (at start (increase (total-cost) (koszt-wodny ?from ?to))))
  )
)