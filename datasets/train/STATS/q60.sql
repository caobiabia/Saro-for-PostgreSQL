select  count(*) from comments as c,          votes as v,          users as u where u.Id = c.UserId 	and c.PostId = v.PostId  AND u.Reputation>=1  AND u.UpVotes=3;
