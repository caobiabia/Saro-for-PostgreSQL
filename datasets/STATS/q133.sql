select  count(*) from comments as c,  		posts as p,  		postHistory as ph,          votes as v,          users as u  where u.Id = c.UserId 	and c.UserId = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = v.UserId  AND p.Score>=-1  AND p.AnswerCount<=6  AND p.CreationDate<='2014-09-08 10:31:11'::timestamp  AND u.Reputation=119  AND u.Views<=199  AND u.CreationDate<='2014-08-31 14:38:43'::timestamp  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp  AND v.CreationDate<='2014-09-09 00:00:00'::timestamp;